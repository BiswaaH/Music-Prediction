import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('spotify-2023.csv', encoding='latin-1')

print("=== ORIGINAL DATASET INFO ===")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== NULL VALUES CHECK ===")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0])

print("\n=== DUPLICATE VALUES CHECK ===")
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

print("\n=== UNIQUE VALUES IN KEY COLUMNS ===")
categorical_cols = ['key', 'mode']
for col in categorical_cols:
    if col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")
        print(f"Values: {df[col].unique()}")

# Data Cleaning Process
print("\n" + "="*50)
print("STARTING DATA CLEANING PROCESS")
print("="*50)

# 1. Handle missing values - replace NaN with 0
print("\n1. Replacing NaN values with 0...")
df_cleaned = df.fillna(0)

# 2. Remove commas from numeric columns and convert to integers
print("\n2. Removing commas and converting to integers...")

# Identify columns that might have commas
numeric_cols = ['streams', 'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists', 
                'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts']

for col in numeric_cols:
    if col in df_cleaned.columns:
        # Convert to string first, remove commas, then convert to numeric
        df_cleaned[col] = df_cleaned[col].astype(str).str.replace(',', '').str.replace('"', '')
        # Convert to numeric, errors='coerce' will turn non-numeric to NaN
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        # Fill any remaining NaN with 0
        df_cleaned[col] = df_cleaned[col].fillna(0).astype(int)

# 3. Convert percentage columns to integers
print("\n3. Converting percentage columns...")
percentage_cols = [col for col in df_cleaned.columns if '%' in col]
for col in percentage_cols:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0).astype(int)

# 4. Convert other numeric columns
print("\n4. Converting other numeric columns...")
other_numeric_cols = ['artist_count', 'released_year', 'released_month', 'released_day', 'bpm']
for col in other_numeric_cols:
    if col in df_cleaned.columns:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0).astype(int)

# 5. Handle categorical columns
print("\n5. Processing categorical columns...")
# For 'key' and 'mode' columns, replace empty strings with 'Unknown'
if 'key' in df_cleaned.columns:
    df_cleaned['key'] = df_cleaned['key'].replace('', 'Unknown').fillna('Unknown')
if 'mode' in df_cleaned.columns:
    df_cleaned['mode'] = df_cleaned['mode'].replace('', 'Unknown').fillna('Unknown')

# 6. Remove duplicates
print("\n6. Removing duplicate rows...")
df_cleaned = df_cleaned.drop_duplicates()

print("\n=== CLEANED DATASET INFO ===")
print(f"Dataset shape after cleaning: {df_cleaned.shape}")

print("\n=== NULL VALUES AFTER CLEANING ===")
null_counts_after = df_cleaned.isnull().sum()
print(f"Total null values: {null_counts_after.sum()}")

print("\n=== DUPLICATE VALUES AFTER CLEANING ===")
duplicates_after = df_cleaned.duplicated().sum()
print(f"Number of duplicate rows: {duplicates_after}")

print("\n=== DATA TYPES AFTER CLEANING ===")
print(df_cleaned.dtypes)

# Feature Selection - Select most relevant features for music popularity analysis
print("\n" + "="*50)
print("FEATURE SELECTION")
print("="*50)

# Select key features for analysis
selected_features = [
    'track_name', 'artist(s)_name', 'artist_count', 'released_year',
    'streams', 'in_spotify_playlists', 'in_spotify_charts',
    'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists',
    'bpm', 'key', 'mode', 'danceability_%', 'valence_%', 'energy_%',
    'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%'
]

# Keep only columns that exist in the dataset
available_features = [col for col in selected_features if col in df_cleaned.columns]
df_final = df_cleaned[available_features].copy()

print(f"Selected {len(available_features)} features for analysis:")
print(available_features)

# Basic Statistics
print("\n=== BASIC STATISTICS ===")
print(df_final.describe())

# Additional Processing Steps
print("\n" + "="*50)
print("ADDITIONAL PROCESSING")
print("="*50)

# 1. Convert object types to numeric where possible
print("\n1. Converting object types to numeric...")
for col in df_final.columns:
    if df_final[col].dtype == 'object' and col not in ['track_name', 'artist(s)_name', 'key', 'mode']:
        # Try to convert to numeric
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        df_final[col] = df_final[col].fillna(0)
        print(f"  Converted {col} to numeric")

# 2. Encode categorical features (key and mode)
print("\n2. Encoding categorical features...")
le_key = LabelEncoder()
le_mode = LabelEncoder()

if 'key' in df_final.columns:
    df_final['key_encoded'] = le_key.fit_transform(df_final['key'].astype(str))
    print(f"  Encoded 'key' feature: {len(le_key.classes_)} unique values")
    print(f"  Key mapping: {dict(zip(le_key.classes_, range(len(le_key.classes_))))}")

if 'mode' in df_final.columns:
    df_final['mode_encoded'] = le_mode.fit_transform(df_final['mode'].astype(str))
    print(f"  Encoded 'mode' feature: {len(le_mode.classes_)} unique values")
    print(f"  Mode mapping: {dict(zip(le_mode.classes_, range(len(le_mode.classes_))))}")

# 3. Drop artist name columns
print("\n3. Dropping artist name columns...")
columns_to_drop = ['artist(s)_name', 'track_name']
existing_cols_to_drop = [col for col in columns_to_drop if col in df_final.columns]
if existing_cols_to_drop:
    df_final = df_final.drop(columns=existing_cols_to_drop)
    print(f"  Dropped columns: {existing_cols_to_drop}")

# Also drop original categorical columns if encoded versions exist
if 'key_encoded' in df_final.columns and 'key' in df_final.columns:
    df_final = df_final.drop(columns=['key'])
    print("  Dropped original 'key' column")

if 'mode_encoded' in df_final.columns and 'mode' in df_final.columns:
    df_final = df_final.drop(columns=['mode'])
    print("  Dropped original 'mode' column")

print("\n=== FINAL DATASET INFO ===")
print(f"Final dataset shape: {df_final.shape}")
print(f"Final columns: {list(df_final.columns)}")
print("\n=== FINAL DATA TYPES ===")
print(df_final.dtypes)

# Convert any remaining object columns to numeric
for col in df_final.columns:
    if df_final[col].dtype == 'object':
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)

print("\n=== FINAL DATA TYPES AFTER CONVERSION ===")
print(df_final.dtypes)

# Save cleaned dataset
df_final.to_csv('spotify_2023_cleaned.csv', index=False)
print(f"\nFinal cleaned dataset saved as 'spotify_2023_cleaned.csv'")

# Visualization
print("\n" + "="*50)
print("CREATING VISUALIZATIONS")
print("="*50)

plt.figure(figsize=(15, 12))

# 1. Distribution of streams
plt.subplot(2, 3, 1)
plt.hist(df_final['streams'], bins=50, alpha=0.7, color='skyblue')
plt.title('Distribution of Streams')
plt.xlabel('Streams')
plt.ylabel('Frequency')

# 2. Distribution of artist count
plt.subplot(2, 3, 2)
if 'artist_count' in df_final.columns:
    plt.hist(df_final['artist_count'], bins=20, alpha=0.7, color='lightcoral')
    plt.title('Distribution of Artist Count')
    plt.xlabel('Number of Artists')
    plt.ylabel('Frequency')
else:
    plt.text(0.5, 0.5, 'Artist count data not available', ha='center', va='center')
    plt.title('Artist Count Distribution')

# 3. Correlation heatmap of audio features
plt.subplot(2, 3, 3)
audio_features = [col for col in df_final.columns if '%' in col]
if len(audio_features) > 1:
    corr_matrix = df_final[audio_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Audio Features Correlation')

# 4. Streams by release year
plt.subplot(2, 3, 4)
yearly_streams = df_final.groupby('released_year')['streams'].mean()
plt.plot(yearly_streams.index, yearly_streams.values, marker='o')
plt.title('Average Streams by Release Year')
plt.xlabel('Release Year')
plt.ylabel('Average Streams')

# 5. BPM distribution
plt.subplot(2, 3, 5)
plt.hist(df_final['bpm'], bins=30, alpha=0.7, color='lightgreen')
plt.title('BPM Distribution')
plt.xlabel('BPM')
plt.ylabel('Frequency')

# 6. Platform presence comparison
plt.subplot(2, 3, 6)
platform_cols = ['in_spotify_playlists', 'in_apple_playlists', 'in_deezer_playlists']
platform_data = [df_final[col].mean() for col in platform_cols if col in df_final.columns]
platform_names = [col.replace('in_', '').replace('_playlists', '') for col in platform_cols if col in df_final.columns]
plt.bar(platform_names, platform_data, color=['green', 'red', 'orange'])
plt.title('Average Playlist Presence by Platform')
plt.ylabel('Average Playlists')

plt.tight_layout()
plt.savefig('spotify_analysis_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'spotify_analysis_plots.png'")

# Summary Report
print("\n" + "="*50)
print("SUMMARY REPORT")
print("="*50)

print(f"[✓] Dataset successfully cleaned and processed")
print(f"[✓] Original shape: {df.shape}")
print(f"[✓] Final shape: {df_final.shape}")
print(f"[✓] Null values replaced with 0")
print(f"[✓] Commas removed from numeric columns")
print(f"[✓] All values converted to appropriate data types")
print(f"[✓] {duplicates - duplicates_after} duplicate rows removed")
print(f"[✓] Key and mode features encoded")
print(f"[✓] Artist name columns dropped")
print(f"[✓] All object types converted to numeric")
print(f"[✓] Cleaned dataset saved as 'spotify_2023_cleaned.csv'")
print(f"[✓] Analysis plots saved as 'spotify_analysis_plots.png'")

print(f"\nTop 5 most streamed songs (by streams):")
top_songs = df_final.nlargest(5, 'streams')[['streams']]
for i, (idx, row) in enumerate(top_songs.iterrows()):
    print(f"  Song #{i+1} - {row['streams']:,} streams")