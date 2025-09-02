import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Music Popularity Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Fixed text visibility issues
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #191414;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #000000; /* Black box */
        padding: 1rem;
        border-left: 4px solid #1DB954;
        margin: 1rem 0;
        color: #FFFFFF; /* Default text = White */
        border-radius: 0.5rem;
    }
    .insight-box h4 {
        color: #1DB954 !important; /* Green heading */
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .insight-box ul,
    .insight-box li,
    .insight-box p {
        color: #FFFFFF !important; /* White for readability */
    }
    .insight-box strong {
        color: #1DB954 !important; /* Green highlights */
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    df = pd.read_csv('spotify_2023_cleaned.csv')
    return df

@st.cache_resource
def train_model(df):
    """Train the Random Forest model"""
    X = df.drop(['streams'], axis=1)
    y = np.log1p(df['streams'])  # Log transformation
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, importance_df, X_train.columns

def predict_popularity(model, features, feature_names):
    """Make prediction using the trained model"""
    # Create feature array in correct order
    feature_array = np.array([features[col] for col in feature_names]).reshape(1, -1)
    log_prediction = model.predict(feature_array)[0]
    # Convert back from log scale
    prediction = np.expm1(log_prediction)
    return prediction, log_prediction

def main():
    # Load data and train model
    df = load_data()
    model, importance_df, feature_names = train_model(df)
    
    # Main header
    st.markdown('<h1 class="main-header">🎵 Music Popularity Predictor</h1>', unsafe_allow_html=True)
    st.markdown("**Predict song success and get data-driven insights for Marketing & A&R teams**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", 
                               ["🎯 Popularity Predictor", "📊 Business Insights", "📈 Model Analytics", "💡 Recommendations"])
    
    if page == "🎯 Popularity Predictor":
        prediction_page(model, feature_names, df)
    elif page == "📊 Business Insights":
        insights_page(df, importance_df)
    elif page == "📈 Model Analytics":
        analytics_page(df, importance_df)
    elif page == "💡 Recommendations":
        recommendations_page(importance_df)

def prediction_page(model, feature_names, df):
    st.markdown('<h2 class="sub-header">🎯 Song Popularity Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎵 Song Features")
        
        # Platform features
        st.markdown("**Platform Presence**")
        spotify_playlists = st.number_input("Spotify Playlists", min_value=0, max_value=50000, value=1000, step=100)
        spotify_charts = st.number_input("Spotify Charts Position", min_value=0, max_value=200, value=50, step=1)
        apple_playlists = st.number_input("Apple Music Playlists", min_value=0, max_value=1000, value=50, step=10)
        apple_charts = st.number_input("Apple Charts Position", min_value=0, max_value=300, value=100, step=1)
        deezer_playlists = st.number_input("Deezer Playlists", min_value=0, max_value=1000, value=30, step=10)
        
        # Song metadata
        st.markdown("**Song Information**")
        artist_count = st.selectbox("Number of Artists", options=[1, 2, 3, 4, 5, 6, 7, 8], index=0)
        released_year = st.selectbox("Release Year", options=list(range(2020, 2024)), index=3)
        
        # Audio features
        st.markdown("**Audio Characteristics**")
        bpm = st.slider("BPM (Beats Per Minute)", min_value=60, max_value=200, value=120, step=1)
        danceability = st.slider("Danceability %", min_value=0, max_value=100, value=70, step=1)
        valence = st.slider("Valence % (Positivity)", min_value=0, max_value=100, value=60, step=1)
        energy = st.slider("Energy %", min_value=0, max_value=100, value=75, step=1)
        acousticness = st.slider("Acousticness %", min_value=0, max_value=100, value=20, step=1)
        instrumentalness = st.slider("Instrumentalness %", min_value=0, max_value=100, value=5, step=1)
        liveness = st.slider("Liveness %", min_value=0, max_value=100, value=15, step=1)
        speechiness = st.slider("Speechiness %", min_value=0, max_value=100, value=8, step=1)
        
        # Musical key and mode
        key_options = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = st.selectbox("Musical Key", options=key_options, index=0)
        mode = st.selectbox("Mode", options=['Major', 'Minor'], index=0)
        
        # Convert to encoded values
        key_encoded = key_options.index(key) + 1  # +1 because 0 is reserved for 'Unknown'
        mode_encoded = 0 if mode == 'Major' else 1
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Prepare features dictionary
        features = {
            'artist_count': artist_count,
            'released_year': released_year,
            'in_spotify_playlists': spotify_playlists,
            'in_spotify_charts': spotify_charts,
            'in_apple_playlists': apple_playlists,
            'in_apple_charts': apple_charts,
            'in_deezer_playlists': deezer_playlists,
            'bpm': bpm,
            'danceability_%': danceability,
            'valence_%': valence,
            'energy_%': energy,
            'acousticness_%': acousticness,
            'instrumentalness_%': instrumentalness,
            'liveness_%': liveness,
            'speechiness_%': speechiness,
            'key_encoded': key_encoded,
            'mode_encoded': mode_encoded
        }
        
        if st.button("🚀 Predict Popularity", type="primary"):
            prediction, log_prediction = predict_popularity(model, features, feature_names)
            
            # Display prediction
            st.markdown("### 📊 Predicted Streams")
            st.metric("Expected Streams", f"{prediction:,.0f}")
            
            # Success probability
            median_streams = df['streams'].median()
            top_10_percent = df['streams'].quantile(0.9)
            
            if prediction > top_10_percent:
                success_level = "🔥 Viral Hit Potential"
                color = "green"
            elif prediction > median_streams:
                success_level = "⭐ Above Average"
                color = "orange"
            else:
                success_level = "📈 Moderate Success"
                color = "blue"
            
            st.markdown(f"**Success Level:** :{color}[{success_level}]")
            
            # Comparison with dataset
            percentile = (df['streams'] < prediction).mean() * 100
            st.metric("Percentile Rank", f"{percentile:.1f}%")
            
            # Feature impact analysis
            st.markdown("### 🎯 Key Success Factors")
            
            # Calculate feature contributions (simplified)
            top_features = ['in_spotify_playlists', 'released_year', 'artist_count', 'danceability_%', 'energy_%']
            
            for feature in top_features:
                if feature in features:
                    value = features[feature]
                    avg_value = df[feature].mean()
                    if value > avg_value:
                        st.success(f"✅ {feature.replace('_', ' ').title()}: Above average ({value} vs {avg_value:.0f})")
                    else:
                        st.warning(f"⚠️ {feature.replace('_', ' ').title()}: Below average ({value} vs {avg_value:.0f})")
        
        # Quick tips
        st.markdown("### 💡 Quick Tips")
        st.info("**Boost Your Prediction:**\n- Increase Spotify playlist presence\n- Consider collaborations (more artists)\n- Optimize danceability (60-80%)\n- Balance energy and valence")

def insights_page(df, importance_df):
    st.markdown('<h2 class="sub-header">📊 Business Insights Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Songs", f"{len(df):,}")
    with col2:
        st.metric("Avg Streams", f"{df['streams'].mean():,.0f}")
    with col3:
        st.metric("Top Song Streams", f"{df['streams'].max():,.0f}")
    with col4:
        st.metric("Success Rate", f"{(df['streams'] > df['streams'].median()).mean()*100:.1f}%")
    
    # Feature importance visualization
    st.subheader("🎯 Feature Importance Analysis")
    
    fig = px.bar(
        importance_df.head(10), 
        x='importance', 
        y='feature',
        orientation='h',
        title="Top 10 Most Important Features for Predicting Success",
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Platform analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎵 Platform Performance")
        platform_cols = ['in_spotify_playlists', 'in_apple_playlists', 'in_deezer_playlists']
        platform_corr = df[platform_cols + ['streams']].corr()['streams'].drop('streams')
        
        fig = px.bar(
            x=platform_corr.index,
            y=platform_corr.values,
            title="Platform Correlation with Streams",
            labels={'x': 'Platform', 'y': 'Correlation'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎼 Audio Features Impact")
        audio_features = ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%']
        audio_corr = df[audio_features + ['streams']].corr()['streams'].drop('streams')
        
        fig = px.bar(
            x=audio_corr.index,
            y=audio_corr.values,
            title="Audio Features Correlation with Streams",
            labels={'x': 'Audio Feature', 'y': 'Correlation'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Success patterns
    st.subheader("📈 Success Patterns")
    
    # Top vs average comparison
    top_100 = df.nlargest(100, 'streams')
    comparison_features = ['artist_count', 'danceability_%', 'energy_%', 'valence_%']
    
    comparison_data = []
    for feature in comparison_features:
        comparison_data.append({
            'Feature': feature.replace('_%', '').replace('_', ' ').title(),
            'Top 100 Songs': top_100[feature].mean(),
            'All Songs': df[feature].mean()
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig = px.bar(
        comparison_df.melt(id_vars='Feature', var_name='Group', value_name='Value'),
        x='Feature',
        y='Value',
        color='Group',
        barmode='group',
        title="Top 100 Songs vs All Songs - Feature Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

def analytics_page(df, importance_df):
    st.markdown('<h2 class="sub-header">📈 Model Analytics</h2>', unsafe_allow_html=True)
    
    # Model performance metrics
    st.subheader("🎯 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Random Forest R²", "79.0%", "Best Model")
    with col2:
        st.metric("XGBoost R²", "62.5%", "Good Performance")
    with col3:
        st.metric("Linear Regression R²", "47.0%", "Baseline")
    
    # Feature categories
    st.subheader("📊 Feature Category Analysis")
    
    audio_features = ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 
                     'instrumentalness_%', 'liveness_%', 'speechiness_%', 'bpm', 'key_encoded', 'mode_encoded']
    platform_features = ['in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists', 
                         'in_apple_charts', 'in_deezer_playlists']
    other_features = ['artist_count', 'released_year']
    
    # Calculate category importance
    audio_importance = importance_df[importance_df['feature'].isin(audio_features)]['importance'].sum()
    platform_importance = importance_df[importance_df['feature'].isin(platform_features)]['importance'].sum()
    other_importance = importance_df[importance_df['feature'].isin(other_features)]['importance'].sum()
    
    category_data = {
        'Category': ['Platform Features', 'Other Features', 'Audio Features'],
        'Importance': [platform_importance, other_importance, audio_importance],
        'Percentage': [platform_importance*100, other_importance*100, audio_importance*100]
    }
    
    fig = px.pie(
        values=category_data['Importance'],
        names=category_data['Category'],
        title="Feature Category Importance Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed feature analysis
    st.subheader("🔍 Detailed Feature Analysis")
    
    # Interactive feature importance
    n_features = st.slider("Number of top features to display", 5, 17, 10)
    
    fig = px.bar(
        importance_df.head(n_features),
        x='feature',
        y='importance',
        title=f"Top {n_features} Feature Importance",
        color='importance',
        color_continuous_scale='Blues'
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.subheader("🔗 Feature Correlations")
    
    # Select key features for correlation
    key_features = importance_df.head(8)['feature'].tolist() + ['streams']
    corr_matrix = df[key_features].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Correlation Matrix of Key Features",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)

def recommendations_page(importance_df):
    st.markdown('<h2 class="sub-header">💡 Strategic Recommendations</h2>', unsafe_allow_html=True)
    
    # Marketing recommendations
    st.subheader("📢 For Marketing Teams")
    
    st.markdown("""
    <div class="insight-box">
    <h4>🎯 Priority Actions</h4>
    <ul>
        <li><strong>Playlist Strategy (51% impact):</strong> Focus 60% of promotion budget on Spotify playlist inclusion</li>
        <li><strong>Cross-Platform Presence:</strong> Maintain visibility on Deezer and Apple Music</li>
        <li><strong>Timing Strategy:</strong> Leverage recency bias - recent releases perform better</li>
        <li><strong>Chart Positioning:</strong> Aim for chart placements to boost algorithmic promotion</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # A&R recommendations
    st.subheader("🎵 For A&R Teams")
    
    st.markdown("""
    <div class="insight-box">
    <h4>🎼 Artist Development</h4>
    <ul>
        <li><strong>Collaboration Strategy (11% impact):</strong> Encourage multi-artist features</li>
        <li><strong>Audio Optimization:</strong> Guide artists on danceability (60-80% sweet spot)</li>
        <li><strong>Energy Balance:</strong> Maintain 70-85% energy for mainstream appeal</li>
        <li><strong>Portfolio Strategy:</strong> Balance audio characteristics across roster</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Data-driven insights
    st.subheader("📊 Data-Driven Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔥 Success Formula:**
        - Platform Presence (64%)
        - Timing & Collaboration (21%)
        - Audio Features (15%)
        
        **💡 Key Findings:**
        - Spotify playlists = #1 success factor
        - Recent releases have built-in advantage
        - Multi-artist songs perform better
        - Audio features support but don't drive success
        """)
    
    with col2:
        st.markdown("""
        **🎯 Optimization Targets:**
        - Spotify playlist count: 1000+
        - Danceability: 60-80%
        - Energy level: 70-85%
        - Artist collaborations: 2-3 artists
        
        **⚠️ Common Mistakes:**
        - Over-focusing on audio perfection
        - Ignoring platform relationships
        - Poor release timing
        - Underestimating collaboration impact
        """)
    
    # Action plan
    st.subheader("🚀 90-Day Action Plan")
    
    st.markdown("""
    <div class="insight-box">
    <h4>📅 Implementation Roadmap</h4>
    
    <p><strong>Month 1: Foundation</strong></p>
    <ul>
        <li>Audit current playlist relationships</li>
        <li>Analyze competitor platform presence</li>
        <li>Establish baseline audio feature guidelines</li>
    </ul>
    
    <p><strong>Month 2: Strategy</strong></p>
    <ul>
        <li>Develop playlist promotion campaigns</li>
        <li>Plan strategic artist collaborations</li>
        <li>Optimize release calendar timing</li>
    </ul>
    
    <p><strong>Month 3: Execution</strong></p>
    <ul>
        <li>Launch playlist inclusion initiatives</li>
        <li>Execute collaboration projects</li>
        <li>Monitor and adjust based on performance</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # ROI Calculator
    st.subheader("💰 ROI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Investment Scenarios:**")
        playlist_budget = st.number_input("Playlist Promotion Budget ($)", value=10000, step=1000)
        collab_budget = st.number_input("Collaboration Budget ($)", value=5000, step=1000)
        
    with col2:
        st.markdown("**Expected Returns:**")
        # Simplified ROI calculation based on feature importance
        playlist_roi = playlist_budget * 0.51 * 2.5  # 51% importance * 2.5x multiplier
        collab_roi = collab_budget * 0.11 * 3.0     # 11% importance * 3.0x multiplier
        
        st.metric("Playlist Promotion ROI", f"${playlist_roi:,.0f}")
        st.metric("Collaboration ROI", f"${collab_roi:,.0f}")
        st.metric("Total Expected Return", f"${playlist_roi + collab_roi:,.0f}")

if __name__ == "__main__":
    main()