import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load cleaned dataset
df = pd.read_csv('spotify_2023_cleaned.csv')

print("=== EXPLORATORY DATA ANALYSIS ===")
print(f"Dataset shape: {df.shape}")
print(f"\nBasic statistics:")
print(df.describe())

# EDA Visualizations
plt.figure(figsize=(20, 15))

# 1. Streams distribution
plt.subplot(3, 4, 1)
plt.hist(df['streams'], bins=50, alpha=0.7, color='skyblue')
plt.title('Distribution of Streams')
plt.xlabel('Streams')
plt.ylabel('Frequency')

# 2. Log-transformed streams
plt.subplot(3, 4, 2)
log_streams = np.log1p(df['streams'])
plt.hist(log_streams, bins=50, alpha=0.7, color='lightgreen')
plt.title('Log-transformed Streams')
plt.xlabel('Log(Streams + 1)')
plt.ylabel('Frequency')

# 3. Correlation heatmap
plt.subplot(3, 4, 3)
audio_features = ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 
                 'instrumentalness_%', 'liveness_%', 'speechiness_%']
corr_matrix = df[audio_features + ['streams']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Audio Features Correlation')

# 4. Platform presence vs streams
plt.subplot(3, 4, 4)
platform_cols = ['in_spotify_playlists', 'in_apple_playlists', 'in_deezer_playlists']
for col in platform_cols:
    plt.scatter(df[col], df['streams'], alpha=0.5, s=10, label=col.replace('in_', '').replace('_playlists', ''))
plt.xlabel('Playlist Count')
plt.ylabel('Streams')
plt.title('Platform Presence vs Streams')
plt.legend()
plt.yscale('log')

# 5. BPM vs Streams
plt.subplot(3, 4, 5)
plt.scatter(df['bpm'], df['streams'], alpha=0.6, s=15)
plt.xlabel('BPM')
plt.ylabel('Streams')
plt.title('BPM vs Streams')
plt.yscale('log')

# 6. Release year distribution
plt.subplot(3, 4, 6)
year_counts = df['released_year'].value_counts().sort_index()
plt.bar(year_counts.index, year_counts.values, alpha=0.7)
plt.title('Songs by Release Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)

# 7. Energy vs Danceability
plt.subplot(3, 4, 7)
plt.scatter(df['energy_%'], df['danceability_%'], c=np.log1p(df['streams']), 
           cmap='viridis', alpha=0.6, s=15)
plt.colorbar(label='Log(Streams)')
plt.xlabel('Energy %')
plt.ylabel('Danceability %')
plt.title('Energy vs Danceability (colored by streams)')

# 8. Artist count distribution
plt.subplot(3, 4, 8)
artist_counts = df['artist_count'].value_counts().sort_index()
plt.bar(artist_counts.index, artist_counts.values, alpha=0.7, color='orange')
plt.title('Distribution of Artist Count')
plt.xlabel('Number of Artists')
plt.ylabel('Count')

# 9. Top streaming songs by audio features
plt.subplot(3, 4, 9)
top_songs = df.nlargest(100, 'streams')
features_mean = top_songs[audio_features].mean()
plt.bar(range(len(features_mean)), features_mean.values, alpha=0.7, color='red')
plt.xticks(range(len(features_mean)), [f.replace('_%', '') for f in features_mean.index], rotation=45)
plt.title('Audio Features of Top 100 Songs')
plt.ylabel('Average %')

# 10. Valence vs Streams by Mode
plt.subplot(3, 4, 10)
for mode in df['mode_encoded'].unique():
    mode_data = df[df['mode_encoded'] == mode]
    mode_name = 'Major' if mode == 0 else 'Minor'
    plt.scatter(mode_data['valence_%'], mode_data['streams'], alpha=0.6, s=10, label=mode_name)
plt.xlabel('Valence %')
plt.ylabel('Streams')
plt.title('Valence vs Streams by Mode')
plt.legend()
plt.yscale('log')

# 11. Speechiness distribution
plt.subplot(3, 4, 11)
plt.hist(df['speechiness_%'], bins=30, alpha=0.7, color='purple')
plt.title('Speechiness Distribution')
plt.xlabel('Speechiness %')
plt.ylabel('Frequency')

# 12. Platform correlation
plt.subplot(3, 4, 12)
platform_corr = df[platform_cols + ['streams']].corr()
sns.heatmap(platform_corr, annot=True, cmap='Blues', fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Platform Correlation')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== MACHINE LEARNING ANALYSIS ===")

# Prepare features and target
X = df.drop(['streams'], axis=1)
y = df['streams']

# Log transformation of target variable
y_log = np.log1p(y)
print(f"Original streams range: {y.min():,.0f} - {y.max():,.0f}")
print(f"Log-transformed range: {y_log.min():.2f} - {y_log.max():.2f}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

# Model training and evaluation
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

results = {}
feature_importance = {}

print("\n=== MODEL PERFORMANCE ===")
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
    
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # Feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance[name] = importance_df

# Feature Importance Analysis
print("\n=== FEATURE IMPORTANCE ANALYSIS ===")

plt.figure(figsize=(15, 10))

# Random Forest Feature Importance
plt.subplot(2, 2, 1)
rf_importance = feature_importance['Random Forest'].head(10)
plt.barh(range(len(rf_importance)), rf_importance['importance'])
plt.yticks(range(len(rf_importance)), rf_importance['feature'])
plt.title('Random Forest - Top 10 Features')
plt.xlabel('Importance')

# XGBoost Feature Importance
plt.subplot(2, 2, 2)
xgb_importance = feature_importance['XGBoost'].head(10)
plt.barh(range(len(xgb_importance)), xgb_importance['importance'])
plt.yticks(range(len(xgb_importance)), xgb_importance['feature'])
plt.title('XGBoost - Top 10 Features')
plt.xlabel('Importance')

# Combined importance comparison
plt.subplot(2, 2, 3)
combined_importance = pd.merge(
    feature_importance['Random Forest'][['feature', 'importance']].rename(columns={'importance': 'RF_importance'}),
    feature_importance['XGBoost'][['feature', 'importance']].rename(columns={'importance': 'XGB_importance'}),
    on='feature'
)
plt.scatter(combined_importance['RF_importance'], combined_importance['XGB_importance'], alpha=0.7)
plt.xlabel('Random Forest Importance')
plt.ylabel('XGBoost Importance')
plt.title('Feature Importance Comparison')
for i, txt in enumerate(combined_importance['feature'][:10]):
    plt.annotate(txt, (combined_importance['RF_importance'].iloc[i], combined_importance['XGB_importance'].iloc[i]), 
                fontsize=8, alpha=0.7)

# Model performance comparison
plt.subplot(2, 2, 4)
model_names = list(results.keys())
r2_scores = [results[name]['R²'] for name in model_names]
plt.bar(model_names, r2_scores, alpha=0.7, color=['blue', 'green', 'red'])
plt.title('Model Performance (R² Score)')
plt.ylabel('R² Score')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Detailed Feature Analysis
print("\n=== TOP FEATURES ANALYSIS ===")

# Get top 5 features from each model
rf_top5 = feature_importance['Random Forest'].head(5)
xgb_top5 = feature_importance['XGBoost'].head(5)

print("\nRandom Forest Top 5 Features:")
for idx, row in rf_top5.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print("\nXGBoost Top 5 Features:")
for idx, row in xgb_top5.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Categorize features
audio_features = ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 
                 'instrumentalness_%', 'liveness_%', 'speechiness_%', 'bpm', 'key_encoded', 'mode_encoded']
platform_features = ['in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists', 
                     'in_apple_charts', 'in_deezer_playlists']
other_features = ['artist_count', 'released_year']

# Analyze feature categories
print("\n=== FEATURE CATEGORY ANALYSIS ===")
for model_name, importance_df in feature_importance.items():
    print(f"\n{model_name} - Feature Category Importance:")
    
    audio_importance = importance_df[importance_df['feature'].isin(audio_features)]['importance'].sum()
    platform_importance = importance_df[importance_df['feature'].isin(platform_features)]['importance'].sum()
    other_importance = importance_df[importance_df['feature'].isin(other_features)]['importance'].sum()
    
    print(f"  Audio Features: {audio_importance:.4f} ({audio_importance*100:.1f}%)")
    print(f"  Platform Features: {platform_importance:.4f} ({platform_importance*100:.1f}%)")
    print(f"  Other Features: {other_importance:.4f} ({other_importance*100:.1f}%)")

print("\n=== BUSINESS INSIGHTS ===")
print("\n🎵 MARKETING & A&R RECOMMENDATIONS:")

# Top features analysis
all_top_features = set(rf_top5['feature'].tolist() + xgb_top5['feature'].tolist())

print(f"\n📊 KEY SUCCESS FACTORS:")
for feature in all_top_features:
    if feature in platform_features:
        print(f"  • {feature}: Platform presence is crucial for visibility")
    elif feature in audio_features:
        feature_clean = feature.replace('_%', '').replace('_encoded', '')
        print(f"  • {feature_clean}: Audio characteristic impacts listener engagement")
    elif feature == 'artist_count':
        print(f"  • Artist collaborations: Multiple artists can boost popularity")
    elif feature == 'released_year':
        print(f"  • Release timing: Recent releases have advantage")

print(f"\n🎯 ACTIONABLE STRATEGIES:")
print("  1. PLAYLIST STRATEGY: Focus on getting songs into major playlists")
print("  2. CROSS-PLATFORM: Ensure presence across Spotify, Apple, and Deezer")
print("  3. AUDIO OPTIMIZATION: Balance energy, danceability, and valence")
print("  4. COLLABORATION: Consider multi-artist features for broader appeal")
print("  5. TIMING: Leverage current music trends and release patterns")

print(f"\n📈 PRACTICAL APPLICATIONS:")
print("  • Playlist Curation: Use audio features to match listener preferences")
print("  • Artist Development: Guide artists on optimal song characteristics")
print("  • Marketing Budget: Prioritize platform promotion based on importance")
print("  • A&R Decisions: Evaluate potential based on feature combinations")

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv('model_results.csv')

# Save feature importance
for model_name, importance_df in feature_importance.items():
    importance_df.to_csv(f'{model_name.lower().replace(" ", "_")}_feature_importance.csv', index=False)

print(f"\n📁 FILES SAVED:")
print("  • eda_analysis.png - Exploratory data analysis plots")
print("  • model_analysis.png - Model performance and feature importance")
print("  • model_results.csv - Model performance metrics")
print("  • Feature importance CSV files for each model")

print(f"\n✅ ANALYSIS COMPLETE!")