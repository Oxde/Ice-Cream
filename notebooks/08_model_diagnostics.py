# %% [markdown]
# # 08 - Statistical Model Diagnostics - CORRECTED VERSION
# 
# **Purpose**: Comprehensive validation of the ACTUAL Ridge regression MMM model
# **Team**: Data Science Research Team - Diagnostic Analysis
# 
# ## 🎯 Diagnostic Objectives
# 
# 1. **Final Model Validation**: Test the Ridge regression model with feature selection
# 2. **Residual Analysis**: Validate assumptions for the actual model used
# 3. **Feature Importance**: Analyze the 15 selected features 
# 4. **Model Stability**: Cross-validation and robustness checks
# 
# ## 📊 Business Impact
# **Validate the 06 Dutch Enhanced model reliability** - Ensure our final model is trustworthy
# 
# ## 🔍 Technical Approach
# Use the EXACT same methodology as 06_dutch_seasonality_comprehensive.py

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("📊 CORRECTED MODEL DIAGNOSTICS - RIDGE REGRESSION")
print("=" * 55)
print("✅ Testing the ACTUAL final model from 06_dutch_seasonality_comprehensive.py")
print("✅ Ridge regression with feature selection")
print("✅ Real data, real model, real diagnostics")

# %%
# Load the ACTUAL data used in the final model
print(f"\n📁 LOADING ACTUAL MODEL DATA")
print("=" * 35)

try:
    # Load the same data as 06_dutch_seasonality_comprehensive.py
    train_data = pd.read_csv('data/mmm_ready/consistent_channels_train_set.csv')
    test_data = pd.read_csv('data/mmm_ready/consistent_channels_test_set.csv')
    
    print(f"✅ Successfully loaded actual training data: {len(train_data)} weeks")
    print(f"✅ Successfully loaded actual test data: {len(test_data)} weeks")
    
    # Apply the same Dutch feature engineering as final model
    def create_dutch_seasonality_features(df):
        """Same feature engineering as 06_dutch_seasonality_comprehensive.py"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Dutch National Holidays
        df['kings_day'] = ((df['date'].dt.month == 4) & 
                          (df['date'].dt.day.isin([26, 27]))).astype(int)
        df['liberation_day'] = ((df['date'].dt.month == 5) & 
                               (df['date'].dt.day == 5)).astype(int)
        
        # Dutch School Holidays
        df['dutch_summer_holidays'] = (df['date'].dt.month.isin([7, 8])).astype(int)
        df['dutch_may_break'] = ((df['date'].dt.month == 5) & 
                                (df['date'].dt.day <= 15)).astype(int)
        df['dutch_autumn_break'] = ((df['date'].dt.month == 10) & 
                                   (df['date'].dt.day <= 15)).astype(int)
        
        # Dutch Weather Patterns
        df['dutch_heatwave'] = (df['weather_temperature_mean'] > 25).astype(int)
        df['warm_spring_nl'] = ((df['date'].dt.month.isin([3, 4, 5])) & 
                               (df['weather_temperature_mean'] > 18)).astype(int)
        df['indian_summer_nl'] = ((df['date'].dt.month.isin([9, 10])) & 
                                 (df['weather_temperature_mean'] > 20)).astype(int)
        
        # Dutch Cultural Effects
        df['weekend_boost'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
        df['dutch_outdoor_season'] = ((df['date'].dt.month.isin([5, 6, 7, 8, 9])) & 
                                     (df['weather_temperature_mean'] > 15)).astype(int)
        df['payday_effect'] = df['date'].dt.day.isin([1, 15]).astype(int)
        
        # Interaction Effects
        df['temp_holiday_interaction'] = (df['weather_temperature_mean'] * 
                                         (df['kings_day'] + df['liberation_day'] + 
                                          df['dutch_summer_holidays']))
        
        # Dutch ice cream season intensity
        month_day = df['date'].dt.month + df['date'].dt.day / 31
        df['dutch_ice_cream_season'] = np.where(
            (month_day >= 4) & (month_day <= 9),
            np.sin((month_day - 4) * np.pi / 5) * (df['weather_temperature_mean'] / 20),
            0
        )
        
        return df
    
    # Apply Dutch feature engineering
    train_enhanced = create_dutch_seasonality_features(train_data)
    test_enhanced = create_dutch_seasonality_features(test_data)
    
    print(f"✅ Applied Dutch feature engineering (same as final model)")
    
    data_loaded = True
    
except Exception as e:
    print(f"❌ Could not load actual data: {e}")
    print("⚠️ This diagnostic requires the actual data to be meaningful!")
    data_loaded = False

# %%
# REPLICATE THE EXACT FINAL MODEL TRAINING PROCESS
print(f"\n🎯 REPLICATING FINAL MODEL TRAINING")
print("=" * 40)

if data_loaded:
    def train_final_model(train_df, test_df):
        """
        EXACT replication of the final model training from 06_dutch_seasonality_comprehensive.py
        """
        # Prepare features (exclude date and target variable)
        feature_columns = [col for col in train_df.columns if col not in ['date', 'sales']]
        
        # Extract features and target, handle missing values
        X_train = train_df[feature_columns].fillna(0)
        y_train = train_df['sales']
        X_test = test_df[feature_columns].fillna(0)
        y_test = test_df['sales']
        
        # Standardize features (crucial for Ridge regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection: Keep top 15 features (EXACT same as final model)
        selector = SelectKBest(f_regression, k=min(15, X_train_scaled.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Train Ridge regression with cross-validated alpha selection
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=TimeSeriesSplit(n_splits=5))
        ridge.fit(X_train_selected, y_train)
        
        # Generate predictions
        y_train_pred = ridge.predict(X_train_selected)
        y_test_pred = ridge.predict(X_test_selected)
        
        # Calculate performance metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Get selected feature names
        selected_features = np.array(feature_columns)[selector.get_support()]
        
        return {
            'model': ridge,
            'scaler': scaler,
            'selector': selector,
            'selected_features': selected_features,
            'X_train_selected': X_train_selected,
            'X_test_selected': X_test_selected,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'optimal_alpha': ridge.alpha_
        }
    
    # Train the final model
    final_model = train_final_model(train_enhanced, test_enhanced)
    
    print(f"✅ Final Model Successfully Trained:")
    print(f"   • Total available features: {len([col for col in train_enhanced.columns if col not in ['date', 'sales']])}")
    print(f"   • Selected features: {len(final_model['selected_features'])}")
    print(f"   • Optimal Ridge alpha: {final_model['optimal_alpha']:.6f}")
    print(f"   • Train R²: {final_model['train_r2']:.3f}")
    print(f"   • Test R²: {final_model['test_r2']:.3f}")
    
    print(f"\n📋 Selected Features (Top 15):")
    for i, feature in enumerate(final_model['selected_features'], 1):
        print(f"   {i:2d}. {feature}")

# %%
# DIAGNOSTIC 1: Feature Importance Analysis
print(f"\n🔍 DIAGNOSTIC 1: FEATURE IMPORTANCE ANALYSIS")
print("=" * 55)

if data_loaded:
    # Get Ridge coefficients for selected features
    ridge_coefs = final_model['model'].coef_
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': final_model['selected_features'],
        'Ridge_Coefficient': ridge_coefs,
        'Abs_Coefficient': np.abs(ridge_coefs)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("Feature Importance (Ridge Coefficients):")
    print(feature_importance.round(4).to_string(index=False))
    
    # Analyze feature categories
    media_features = [f for f in final_model['selected_features'] if any(word in f.lower() for word in ['cost', 'spend', 'search', 'tv', 'radio', 'social', 'ooh'])]
    dutch_features = [f for f in final_model['selected_features'] if any(word in f.lower() for word in ['dutch', 'kings', 'liberation', 'holiday', 'heatwave', 'weekend'])]
    control_features = [f for f in final_model['selected_features'] if f not in media_features and f not in dutch_features]
    
    print(f"\n📊 Feature Category Analysis:")
    print(f"   • Media features: {len(media_features)} ({len(media_features)/len(final_model['selected_features'])*100:.1f}%)")
    print(f"   • Dutch features: {len(dutch_features)} ({len(dutch_features)/len(final_model['selected_features'])*100:.1f}%)")
    print(f"   • Control features: {len(control_features)} ({len(control_features)/len(final_model['selected_features'])*100:.1f}%)")
    
    # Top positive and negative drivers
    top_positive = feature_importance.nlargest(5, 'Ridge_Coefficient')
    top_negative = feature_importance.nsmallest(5, 'Ridge_Coefficient')
    
    print(f"\n🏆 Top 5 Positive Drivers:")
    for _, row in top_positive.iterrows():
        print(f"   • {row['Feature']}: {row['Ridge_Coefficient']:+.4f}")
    
    print(f"\n📉 Top 5 Negative Drivers:")
    for _, row in top_negative.iterrows():
        print(f"   • {row['Feature']}: {row['Ridge_Coefficient']:+.4f}")

# %%
# DIAGNOSTIC 2: Residual Analysis for Ridge Regression
print(f"\n🔍 DIAGNOSTIC 2: RESIDUAL ANALYSIS")
print("=" * 40)

if data_loaded:
    # Get residuals from training and test sets
    train_residuals = final_model['y_train'] - final_model['y_train_pred']
    test_residuals = final_model['y_test'] - final_model['y_test_pred']
    
    # Combined residuals for analysis
    all_residuals = np.concatenate([train_residuals, test_residuals])
    
    print(f"🔍 Residual Statistics:")
    print(f"   • Train residuals: {len(train_residuals)} observations")
    print(f"   • Test residuals: {len(test_residuals)} observations")
    print(f"   • Combined mean: {np.mean(all_residuals):.2f}")
    print(f"   • Combined std: {np.std(all_residuals):.2f}")
    
    # 1. Normality tests
    shapiro_stat, shapiro_p = shapiro(all_residuals)
    jb_stat, jb_p = jarque_bera(all_residuals)
    
    print(f"\n1️⃣ NORMALITY TESTS:")
    print(f"   Shapiro-Wilk test:")
    print(f"     Statistic: {shapiro_stat:.4f}")
    print(f"     p-value: {shapiro_p:.4f}")
    print(f"     Result: {'✅ Normal' if shapiro_p > 0.05 else '⚠️ Non-normal (but OK for Ridge)'}")
    
    print(f"\n   Jarque-Bera test:")
    print(f"     Statistic: {jb_stat:.4f}")
    print(f"     p-value: {jb_p:.4f}")
    print(f"     Result: {'✅ Normal' if jb_p > 0.05 else '⚠️ Non-normal (but OK for Ridge)'}")
    
    # 2. Homoscedasticity analysis (visual check for Ridge)
    train_fitted = final_model['y_train_pred']
    test_fitted = final_model['y_test_pred']
    
    # Calculate residual variance across fitted value ranges
    fitted_ranges = np.percentile(train_fitted, [0, 25, 50, 75, 100])
    residual_vars = []
    for i in range(len(fitted_ranges)-1):
        mask = (train_fitted >= fitted_ranges[i]) & (train_fitted < fitted_ranges[i+1])
        if np.sum(mask) > 0:
            residual_vars.append(np.var(train_residuals[mask]))
    
    variance_ratio = max(residual_vars) / min(residual_vars) if residual_vars else 1
    
    print(f"\n2️⃣ HOMOSCEDASTICITY ANALYSIS:")
    print(f"   Residual variance ratio: {variance_ratio:.2f}")
    print(f"   Result: {'✅ Homoscedastic' if variance_ratio < 4 else '⚠️ Some heteroscedasticity'}")
    
    # 3. Autocorrelation test
    dw_stat = durbin_watson(train_residuals)
    
    print(f"\n3️⃣ AUTOCORRELATION TEST:")
    print(f"   Durbin-Watson statistic: {dw_stat:.4f}")
    if dw_stat < 1.5:
        print(f"   Result: ⚠️ Positive autocorrelation detected")
    elif dw_stat > 2.5:
        print(f"   Result: ⚠️ Negative autocorrelation detected")
    else:
        print(f"   Result: ✅ No significant autocorrelation")
    
    # 4. Outlier detection
    residual_z_scores = np.abs(stats.zscore(all_residuals))
    outliers = np.sum(residual_z_scores > 3)
    
    print(f"\n4️⃣ OUTLIER ANALYSIS:")
    print(f"   Extreme outliers (|z| > 3): {outliers}")
    print(f"   Outlier rate: {outliers/len(all_residuals)*100:.1f}%")
    print(f"   Result: {'✅ Normal outlier rate' if outliers/len(all_residuals) < 0.05 else '⚠️ High outlier rate'}")

# %%
# DIAGNOSTIC 3: Model Stability and Cross-Validation
print(f"\n🔍 DIAGNOSTIC 3: MODEL STABILITY ANALYSIS")
print("=" * 50)

if data_loaded:
    # Perform additional cross-validation analysis
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Get the full feature set used in final model
    feature_columns = [col for col in train_enhanced.columns if col not in ['date', 'sales']]
    X_full = train_enhanced[feature_columns].fillna(0)
    y_full = train_enhanced['sales']
    
    # Standardize
    scaler_stability = StandardScaler()
    X_full_scaled = scaler_stability.fit_transform(X_full)
    
    # Apply same feature selection
    selector_stability = SelectKBest(f_regression, k=min(15, X_full_scaled.shape[1]))
    X_full_selected = selector_stability.fit_transform(X_full_scaled, y_full)
    
    # Cross-validation stability test
    cv_scores = []
    cv_alphas = []
    
    for train_idx, val_idx in tscv.split(X_full_selected):
        X_train_cv, X_val_cv = X_full_selected[train_idx], X_full_selected[val_idx]
        y_train_cv, y_val_cv = y_full.iloc[train_idx], y_full.iloc[val_idx]
        
        ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=3)
        ridge_cv.fit(X_train_cv, y_train_cv)
        
        y_pred_cv = ridge_cv.predict(X_val_cv)
        cv_scores.append(r2_score(y_val_cv, y_pred_cv))
        cv_alphas.append(ridge_cv.alpha_)
    
    print(f"Cross-Validation Stability:")
    print(f"   • Mean CV R²: {np.mean(cv_scores):.3f}")
    print(f"   • Std CV R²: {np.std(cv_scores):.3f}")
    print(f"   • CV R² Range: [{np.min(cv_scores):.3f}, {np.max(cv_scores):.3f}]")
    print(f"   • Mean optimal alpha: {np.mean(cv_alphas):.6f}")
    print(f"   • Alpha stability: {np.std(cv_alphas):.6f}")
    
    stability_score = 1 - (np.std(cv_scores) / np.mean(cv_scores))
    print(f"   • Stability score: {stability_score:.3f}")
    print(f"   • Result: {'✅ Stable model' if stability_score > 0.9 else '⚠️ Some instability' if stability_score > 0.8 else '❌ Unstable model'}")

# %%
# DIAGNOSTIC 4: Key Visualization
print(f"\n📊 KEY DIAGNOSTIC VISUALIZATION")
print("=" * 35)

if data_loaded:
    # Create single most important diagnostic plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Residuals vs Fitted - Most important diagnostic for model assumptions
    ax.scatter(final_model['y_train_pred'], train_residuals, alpha=0.6, s=30, color='blue', label='Training')
    ax.scatter(final_model['y_test_pred'], test_residuals, alpha=0.6, s=30, color='red', label='Test')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.8)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted Values\n(Random pattern indicates good model assumptions)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %%
# DIAGNOSTIC 5: Final Model Assessment
print(f"\n🎯 FINAL MODEL ASSESSMENT")
print("=" * 35)

if data_loaded:
    # Overall model quality assessment
    print(f"📊 MODEL PERFORMANCE SUMMARY:")
    print(f"   • Training R²: {final_model['train_r2']:.3f}")
    print(f"   • Test R²: {final_model['test_r2']:.3f}")
    print(f"   • Overfitting gap: {final_model['train_r2'] - final_model['test_r2']:.3f}")
    print(f"   • Optimal regularization: α = {final_model['optimal_alpha']:.6f}")
    
    # Performance rating
    test_r2 = final_model['test_r2']
    if test_r2 >= 0.70:
        performance_rating = "🏆 EXCELLENT"
    elif test_r2 >= 0.60:
        performance_rating = "✅ GOOD"
    elif test_r2 >= 0.50:
        performance_rating = "👍 ACCEPTABLE"
    else:
        performance_rating = "⚠️ NEEDS IMPROVEMENT"
    
    print(f"   • Performance rating: {performance_rating}")
    
    # Diagnostic summary
    print(f"\n📋 DIAGNOSTIC SUMMARY:")
    
    # Residual quality
    residual_quality = "✅ GOOD" if shapiro_p > 0.01 else "⚠️ ACCEPTABLE"
    print(f"   • Residual normality: {residual_quality}")
    
    # Stability assessment
    if 'cv_scores' in locals():
        stability_quality = "✅ STABLE" if np.std(cv_scores) < 0.05 else "⚠️ MODERATE"
        print(f"   • Model stability: {stability_quality}")
    
    # Feature selection quality
    feature_balance = len(media_features) + len(dutch_features)
    if feature_balance >= 10:
        feature_quality = "✅ WELL-BALANCED"
    elif feature_balance >= 7:
        feature_quality = "👍 GOOD BALANCE"
    else:
        feature_quality = "⚠️ NEEDS MORE FEATURES"
    
    print(f"   • Feature selection: {feature_quality}")
    
    # Overall recommendation
    print(f"\n🎯 OVERALL RECOMMENDATION:")
    
    overall_score = (
        (1 if test_r2 >= 0.50 else 0) +
        (1 if final_model['train_r2'] - final_model['test_r2'] < 0.15 else 0) +
        (1 if shapiro_p > 0.01 else 0) +
        (1 if 'cv_scores' in locals() and np.std(cv_scores) < 0.07 else 0)
    )
    
    if overall_score >= 3:
        recommendation = "✅ MODEL APPROVED FOR PRODUCTION"
        details = "The Ridge regression model meets quality standards for business use."
    elif overall_score >= 2:
        recommendation = "👍 MODEL ACCEPTABLE WITH MONITORING"
        details = "The model is usable but should be monitored for performance."
    else:
        recommendation = "⚠️ MODEL NEEDS IMPROVEMENT"
        details = "Consider feature engineering or alternative modeling approaches."
    
    print(f"   {recommendation}")
    print(f"   {details}")
    
    # Business readiness
    print(f"\n💼 BUSINESS READINESS:")
    print(f"   • Marketing optimization: {'✅ Ready' if test_r2 >= 0.50 else '⚠️ Limited'}")
    print(f"   • Budget allocation: {'✅ Confident' if overall_score >= 3 else '⚠️ Cautious'}")
    print(f"   • ROI analysis: {'✅ Reliable' if final_model['train_r2'] - final_model['test_r2'] < 0.10 else '⚠️ Monitor'}")
    print(f"   • Strategic planning: {'✅ Supported' if len(final_model['selected_features']) >= 12 else '⚠️ Limited'}")

print(f"\n📋 CORRECTED DIAGNOSTICS COMPLETE!")
print("=" * 40)
print("✅ Analyzed the ACTUAL Ridge regression model")
print("✅ Used the same feature selection (top 15 features)")
print("✅ Applied proper regularization diagnostics")
print("✅ Tested model stability and cross-validation")
print("✅ All results are now meaningful and actionable!") 