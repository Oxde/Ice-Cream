#!/usr/bin/env python3
"""
Media Mix Model (MMM) - First Iteration
Simple implementation focused on preventing overfitting and business interpretability.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class MediaMixModel:
    def __init__(self, adstock_max_lag=6, regularization_strength=1.0):
        """
        Simple MMM with adstock and saturation transformations.
        
        Args:
            adstock_max_lag: Maximum lag for adstock decay (weeks)
            regularization_strength: Ridge regression alpha parameter
        """
        self.adstock_max_lag = adstock_max_lag
        self.regularization_strength = regularization_strength
        self.scaler = StandardScaler()
        self.model = None
        self.media_channels = None
        self.control_variables = None
        self.adstock_params = {}
        self.saturation_params = {}
        self.feature_names = None
        
    def adstock_transform(self, x, decay_rate):
        """
        Apply adstock (carryover) transformation to media spend.
        
        Args:
            x: Media spend series
            decay_rate: Decay rate between 0 and 1
        """
        adstocked = np.zeros_like(x)
        adstocked[0] = x[0]
        
        for i in range(1, len(x)):
            adstocked[i] = x[i] + decay_rate * adstocked[i-1]
            
        return adstocked
    
    def saturation_transform(self, x, alpha, gamma):
        """
        Apply saturation curve transformation (diminishing returns).
        
        Args:
            x: Adstocked media spend
            alpha: Saturation slope parameter
            gamma: Saturation half-saturation point
        """
        return alpha * x / (gamma + x)
    
    def prepare_media_features(self, data, media_channels, optimize_params=True):
        """
        Prepare media features with adstock and saturation transformations.
        """
        transformed_features = {}
        
        for channel in media_channels:
            if channel not in data.columns:
                print(f"Warning: {channel} not found in data")
                continue
                
            # Simple adstock parameters (can be optimized later)
            if optimize_params and len(data) > 50:
                # Use median values as starting point for stability
                decay_rate = 0.3  # Conservative decay
                alpha = np.median(data[channel])  # Scale parameter
                gamma = np.percentile(data[channel], 50)  # Half-saturation point
            else:
                # Fixed parameters for smaller datasets
                decay_rate = 0.2
                alpha = 1.0
                gamma = 1.0
            
            # Apply transformations
            adstocked = self.adstock_transform(data[channel].values, decay_rate)
            saturated = self.saturation_transform(adstocked, alpha, gamma)
            
            # Store parameters
            self.adstock_params[channel] = decay_rate
            self.saturation_params[channel] = {'alpha': alpha, 'gamma': gamma}
            
            transformed_features[f"{channel}_transformed"] = saturated
            
        return pd.DataFrame(transformed_features, index=data.index)
    
    def fit(self, data, target_col='sales'):
        """
        Fit the MMM model.
        
        Args:
            data: Training data
            target_col: Target variable column name
        """
        # Identify media channels (cost/spend columns)
        self.media_channels = [col for col in data.columns 
                              if any(keyword in col.lower() 
                                   for keyword in ['cost', 'spend', 'campaigns'])]
        
        # Control variables (non-media)
        self.control_variables = [col for col in data.columns 
                                if col not in self.media_channels + [target_col, 'date']]
        
        print(f"Media channels: {self.media_channels}")
        print(f"Control variables: {self.control_variables}")
        
        # Prepare features
        media_features = self.prepare_media_features(data, self.media_channels)
        control_features = data[self.control_variables]
        
        # Combine all features
        X = pd.concat([media_features, control_features], axis=1)
        y = data[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features to prevent issues with different scales
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Ridge regression with cross-validation for regularization
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        self.model = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
        self.model.fit(X_scaled, y)
        
        print(f"Best regularization alpha: {self.model.alpha_}")
        print(f"Model fitted with {X.shape[1]} features on {X.shape[0]} samples")
        
        return self
    
    def predict(self, data):
        """Make predictions on new data."""
        # Prepare features same way as training
        media_features = self.prepare_media_features(data, self.media_channels, optimize_params=False)
        control_features = data[self.control_variables]
        
        X = pd.concat([media_features, control_features], axis=1)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def evaluate(self, data, target_col='sales'):
        """Evaluate model performance."""
        y_true = data[target_col]
        y_pred = self.predict(data)
        
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        return metrics, y_pred
    
    def get_channel_contributions(self, data):
        """
        Calculate contribution of each media channel to sales.
        This is crucial for budget allocation decisions.
        """
        # Prepare features
        media_features = self.prepare_media_features(data, self.media_channels, optimize_params=False)
        control_features = data[self.control_variables]
        
        X = pd.concat([media_features, control_features], axis=1)
        X_scaled = self.scaler.transform(X)
        
        # Get model coefficients
        coefficients = self.model.coef_
        
        # Calculate contributions
        contributions = {}
        base_prediction = np.full(len(data), self.model.intercept_)
        
        # Media contributions
        for i, channel in enumerate(self.media_channels):
            feature_name = f"{channel}_transformed"
            if feature_name in self.feature_names:
                feature_idx = self.feature_names.index(feature_name)
                contribution = X_scaled[:, feature_idx] * coefficients[feature_idx]
                contributions[channel] = contribution
        
        # Control contributions (grouped)
        control_contribution = np.zeros(len(data))
        for i, var in enumerate(self.control_variables):
            if var in self.feature_names:
                feature_idx = self.feature_names.index(var)
                control_contribution += X_scaled[:, feature_idx] * coefficients[feature_idx]
        
        contributions['Control_Variables'] = control_contribution
        contributions['Base'] = base_prediction
        
        return contributions
    
    def plot_model_performance(self, train_data, test_data, target_col='sales'):
        """Plot model performance and diagnostics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training performance
        train_metrics, train_pred = self.evaluate(train_data, target_col)
        test_metrics, test_pred = self.evaluate(test_data, target_col)
        
        # 1. Actual vs Predicted - Training
        axes[0, 0].scatter(train_data[target_col], train_pred, alpha=0.6, label='Training')
        axes[0, 0].plot([train_data[target_col].min(), train_data[target_col].max()], 
                       [train_data[target_col].min(), train_data[target_col].max()], 'r--')
        axes[0, 0].set_xlabel('Actual Sales')
        axes[0, 0].set_ylabel('Predicted Sales')
        axes[0, 0].set_title(f'Training: R² = {train_metrics["R2"]:.3f}')
        
        # 2. Actual vs Predicted - Test
        axes[0, 1].scatter(test_data[target_col], test_pred, alpha=0.6, label='Test', color='orange')
        axes[0, 1].plot([test_data[target_col].min(), test_data[target_col].max()], 
                       [test_data[target_col].min(), test_data[target_col].max()], 'r--')
        axes[0, 1].set_xlabel('Actual Sales')
        axes[0, 1].set_ylabel('Predicted Sales')
        axes[0, 1].set_title(f'Test: R² = {test_metrics["R2"]:.3f}')
        
        # 3. Time series plot
        combined_data = pd.concat([train_data, test_data])
        combined_pred = np.concatenate([train_pred, test_pred])
        
        axes[1, 0].plot(combined_data['date'], combined_data[target_col], 
                       label='Actual', linewidth=2)
        axes[1, 0].plot(combined_data['date'], combined_pred, 
                       label='Predicted', linewidth=2, alpha=0.8)
        axes[1, 0].axvline(x=train_data['date'].iloc[-1], color='red', 
                          linestyle='--', label='Train/Test Split')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Sales')
        axes[1, 0].set_title('Time Series: Actual vs Predicted')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Feature importance (coefficients)
        coefficients = self.model.coef_
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': coefficients
        }).sort_values('Coefficient', key=abs, ascending=False).head(10)
        
        axes[1, 1].barh(feature_importance['Feature'], feature_importance['Coefficient'])
        axes[1, 1].set_xlabel('Coefficient Value')
        axes[1, 1].set_title('Top 10 Feature Importance')
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics
        print("\n=== MODEL PERFORMANCE ===")
        print("Training Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.3f}")
    
    def get_budget_allocation_insights(self, data):
        """
        Provide budget allocation insights for business decisions.
        """
        contributions = self.get_channel_contributions(data)
        
        # Calculate total contribution by channel
        channel_summary = {}
        for channel in self.media_channels:
            if channel in contributions:
                total_contribution = np.sum(contributions[channel])
                avg_spend = data[channel].mean()
                efficiency = total_contribution / avg_spend if avg_spend > 0 else 0
                
                channel_summary[channel] = {
                    'Total_Contribution': total_contribution,
                    'Avg_Spend': avg_spend,
                    'Efficiency': efficiency
                }
        
        # Sort by efficiency
        sorted_channels = sorted(channel_summary.items(), 
                               key=lambda x: x[1]['Efficiency'], 
                               reverse=True)
        
        print("\n=== BUDGET ALLOCATION INSIGHTS ===")
        print("Channel Efficiency Ranking (Contribution per $ spent):")
        for i, (channel, metrics) in enumerate(sorted_channels, 1):
            print(f"{i}. {channel}")
            print(f"   Efficiency: {metrics['Efficiency']:.2f}")
            print(f"   Avg Spend: ${metrics['Avg_Spend']:,.0f}")
            print(f"   Total Contribution: {metrics['Total_Contribution']:,.0f}")
            print()
        
        return channel_summary


def main():
    """Main function to run the MMM analysis."""
    
    # Load data
    print("Loading data...")
    train_data = pd.read_csv('data/mmm_ready/consistent_channels_train_set.csv')
    test_data = pd.read_csv('data/mmm_ready/consistent_channels_test_set.csv')
    
    # Convert date columns
    train_data['date'] = pd.to_datetime(train_data['date'])
    test_data['date'] = pd.to_datetime(test_data['date'])
    
    print(f"Training data: {train_data.shape[0]} samples")
    print(f"Test data: {test_data.shape[0]} samples")
    
    # Initialize and fit model
    print("\nFitting MMM model...")
    mmm = MediaMixModel(adstock_max_lag=4, regularization_strength=1.0)
    mmm.fit(train_data)
    
    # Evaluate model
    print("\nEvaluating model...")
    mmm.plot_model_performance(train_data, test_data)
    
    # Get business insights
    print("\nGenerating business insights...")
    mmm.get_budget_allocation_insights(train_data)
    
    return mmm

if __name__ == "__main__":
    model = main() 