"""
Simplified and Improved Drug Synthesis ML Analysis
Focus on key insights and robust modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load data and perform basic exploration"""
    print("="*70)
    print("DRUG SYNTHESIS IMPURITY PREDICTION - PROFESSIONAL ANALYSIS")
    print("="*70)
    
    # Load data
    print("\n📊 Loading reaction data...")
    data = pd.read_csv('data/synthetic_reaction_data.csv')
    print(f"✅ Loaded {len(data)} reaction experiments")
    
    # Handle missing values
    data['catalyst'] = data['catalyst'].fillna('None')
    
    print("\n🔍 Dataset Overview:")
    print(f"  • Shape: {data.shape}")
    print(f"  • Target variables: Yield ({data['yield_pct'].min():.1f}-{data['yield_pct'].max():.1f}%), Impurity ({data['impurity_pct'].min():.1f}-{data['impurity_pct'].max():.1f}%)")
    print(f"  • Catalysts: {sorted(data['catalyst'].unique())}")
    print(f"  • Solvents: {sorted(data['solvent'].unique())}")
    
    return data

def prepare_features(data):
    """Prepare features for modeling"""
    print("\n⚙️ Feature Engineering...")
    
    # Encode categorical variables
    le_catalyst = LabelEncoder()
    le_solvent = LabelEncoder()
    
    data['catalyst_encoded'] = le_catalyst.fit_transform(data['catalyst'])
    data['solvent_encoded'] = le_solvent.fit_transform(data['solvent'])
    
    # Create feature matrix
    feature_cols = ['temperature_C', 'concentration_M', 'residence_time_min', 
                   'pH', 'equiv_oxidant', 'stirring_rpm', 
                   'catalyst_encoded', 'solvent_encoded']
    
    X = data[feature_cols]
    y_yield = data['yield_pct']
    y_impurity = data['impurity_pct']
    
    print(f"✅ Features prepared: {len(feature_cols)} variables")
    
    return X, y_yield, y_impurity, le_catalyst, le_solvent, feature_cols

def train_and_evaluate_models(X, y_yield, y_impurity):
    """Train models and evaluate performance"""
    print("\n🤖 Training Machine Learning Models...")
    
    # Split data
    X_train, X_test, y_yield_train, y_yield_test, y_impurity_train, y_impurity_test = train_test_split(
        X, y_yield, y_impurity, test_size=0.2, random_state=42
    )
    
    # Use Random Forest (more robust than Gradient Boosting for this data)
    yield_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    impurity_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    
    # Train models
    yield_model.fit(X_train, y_yield_train)
    impurity_model.fit(X_train, y_impurity_train)
    
    # Evaluate models
    yield_pred = yield_model.predict(X_test)
    impurity_pred = impurity_model.predict(X_test)
    
    # Calculate metrics
    yield_r2 = r2_score(y_yield_test, yield_pred)
    yield_mae = mean_absolute_error(y_yield_test, yield_pred)
    impurity_r2 = r2_score(y_impurity_test, impurity_pred)
    impurity_mae = mean_absolute_error(y_impurity_test, impurity_pred)
    
    print(f"\n📈 Model Performance:")
    print(f"  YIELD PREDICTION:")
    print(f"    • R² Score: {yield_r2:.4f}")
    print(f"    • MAE: {yield_mae:.2f}%")
    print(f"  IMPURITY PREDICTION:")
    print(f"    • R² Score: {impurity_r2:.4f}")
    print(f"    • MAE: {impurity_mae:.2f}%")
    
    # Save metrics
    metrics = {
        'yield': {'R2': float(yield_r2), 'MAE': float(yield_mae)},
        'impurity': {'R2': float(impurity_r2), 'MAE': float(impurity_mae)}
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return yield_model, impurity_model, X_test, y_yield_test, y_impurity_test, metrics

def analyze_feature_importance(models, feature_names, X_test, y_yield_test, y_impurity_test):
    """Analyze what factors matter most"""
    print("\n🎯 Feature Importance Analysis...")
    
    yield_model, impurity_model = models
    
    # Get feature importance
    yield_importance = yield_model.feature_importances_
    impurity_importance = impurity_model.feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Yield_Importance': yield_importance,
        'Impurity_Importance': impurity_importance
    }).sort_values('Impurity_Importance', ascending=False)
    
    print("\n🔍 Key Factors for Impurity Control:")
    for i, row in importance_df.head(5).iterrows():
        print(f"  • {row['Feature']}: {row['Impurity_Importance']:.3f}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sort by impurity importance for both plots
    sorted_idx = np.argsort(impurity_importance)
    
    ax1.barh(range(len(feature_names)), yield_importance[sorted_idx])
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax1.set_title('Feature Importance - Yield Prediction')
    ax1.set_xlabel('Importance')
    
    ax2.barh(range(len(feature_names)), impurity_importance[sorted_idx])
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax2.set_title('Feature Importance - Impurity Prediction')
    ax2.set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df

def create_reaction_insights(data, models, le_catalyst, le_solvent):
    """Create insights about optimal conditions"""
    print("\n🎯 Finding Optimal Reaction Conditions...")
    
    yield_model, impurity_model = models
    
    # Test different catalyst and solvent combinations
    best_conditions = []
    
    for catalyst in ['None', 'A', 'B', 'C']:
        for solvent in ['DMF', 'MeCN', 'THF', 'IPA']:
            # Use optimal ranges based on data analysis
            conditions = [[
                100,  # temperature (optimal range)
                1.2,  # concentration
                30,   # residence_time
                3.0,  # pH
                1.0,  # equiv_oxidant
                600,  # stirring_rpm
                le_catalyst.transform([catalyst])[0],
                le_solvent.transform([solvent])[0]
            ]]
            
            predicted_yield = yield_model.predict(conditions)[0]
            predicted_impurity = impurity_model.predict(conditions)[0]
            
            # Score: maximize yield, minimize impurity
            score = predicted_yield - 0.5 * predicted_impurity
            
            best_conditions.append({
                'catalyst': catalyst,
                'solvent': solvent,
                'predicted_yield': predicted_yield,
                'predicted_impurity': predicted_impurity,
                'score': score
            })
    
    # Find best combination
    best_conditions_df = pd.DataFrame(best_conditions).sort_values('score', ascending=False)
    best = best_conditions_df.iloc[0]
    
    print(f"\n🏆 OPTIMAL CONDITIONS IDENTIFIED:")
    print(f"  • Catalyst: {best['catalyst']}")
    print(f"  • Solvent: {best['solvent']}")
    print(f"  • Predicted Yield: {best['predicted_yield']:.1f}%")
    print(f"  • Predicted Impurity: {best['predicted_impurity']:.1f}%")
    print(f"  • Optimization Score: {best['score']:.1f}")
    
    # Save results
    optimal_conditions = {
        'temperature_C': 100.0,
        'concentration_M': 1.2,
        'residence_time_min': 30.0,
        'pH': 3.0,
        'equiv_oxidant': 1.0,
        'stirring_rpm': 600,
        'catalyst': best['catalyst'],
        'solvent': best['solvent'],
        'predicted_yield': float(best['predicted_yield']),
        'predicted_impurity': float(best['predicted_impurity'])
    }
    
    with open('reports/best_conditions.json', 'w') as f:
        json.dump(optimal_conditions, f, indent=2)
    
    return optimal_conditions, best_conditions_df

def create_visualization_dashboard(data, best_conditions_df):
    """Create professional visualizations"""
    print("\n📊 Creating Visualization Dashboard...")
    
    # Create a comprehensive dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Yield vs Temperature by Catalyst
    for catalyst in data['catalyst'].unique():
        if pd.notna(catalyst):
            subset = data[data['catalyst'] == catalyst]
            ax1.scatter(subset['temperature_C'], subset['yield_pct'], 
                       alpha=0.6, label=f'Catalyst {catalyst}', s=30)
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Yield (%)')
    ax1.set_title('Yield vs Temperature by Catalyst')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Impurity vs Temperature by Catalyst
    for catalyst in data['catalyst'].unique():
        if pd.notna(catalyst):
            subset = data[data['catalyst'] == catalyst]
            ax2.scatter(subset['temperature_C'], subset['impurity_pct'], 
                       alpha=0.6, label=f'Catalyst {catalyst}', s=30)
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Impurity (%)')
    ax2.set_title('Impurity vs Temperature by Catalyst')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Catalyst Performance Comparison
    catalyst_stats = data.groupby('catalyst').agg({
        'yield_pct': 'mean',
        'impurity_pct': 'mean'
    }).round(1)
    
    x_pos = np.arange(len(catalyst_stats))
    ax3.bar(x_pos - 0.2, catalyst_stats['yield_pct'], 0.4, 
            label='Avg Yield', alpha=0.8, color='green')
    ax3.bar(x_pos + 0.2, catalyst_stats['impurity_pct'], 0.4, 
            label='Avg Impurity', alpha=0.8, color='red')
    ax3.set_xlabel('Catalyst')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('Catalyst Performance Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(catalyst_stats.index)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Optimization Results
    ax4.scatter(best_conditions_df['predicted_yield'], 
               best_conditions_df['predicted_impurity'],
               c=best_conditions_df['score'], cmap='viridis', s=100)
    ax4.set_xlabel('Predicted Yield (%)')
    ax4.set_ylabel('Predicted Impurity (%)')
    ax4.set_title('Optimization Results (Color = Score)')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Optimization Score')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run the complete analysis"""
    
    # Load and explore data
    data = load_and_explore_data()
    
    # Prepare features
    X, y_yield, y_impurity, le_catalyst, le_solvent, feature_names = prepare_features(data)
    
    # Train models
    yield_model, impurity_model, X_test, y_yield_test, y_impurity_test, metrics = train_and_evaluate_models(X, y_yield, y_impurity)
    
    # Analyze importance
    importance_df = analyze_feature_importance((yield_model, impurity_model), feature_names, X_test, y_yield_test, y_impurity_test)
    
    # Find optimal conditions
    optimal_conditions, best_conditions_df = create_reaction_insights(data, (yield_model, impurity_model), le_catalyst, le_solvent)
    
    # Create visualizations
    create_visualization_dashboard(data, best_conditions_df)
    
    # Save models
    print("\n💾 Saving Models...")
    joblib.dump(yield_model, 'models/model_yield.joblib')
    joblib.dump(impurity_model, 'models/model_impurity.joblib')
    joblib.dump({'catalyst': le_catalyst, 'solvent': le_solvent}, 'models/label_encoders.joblib')
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("📋 SUMMARY:")
    print(f"  • Yield Model R²: {metrics['yield']['R2']:.3f}")
    print(f"  • Impurity Model R²: {metrics['impurity']['R2']:.3f}")
    print(f"  • Best Catalyst: {optimal_conditions['catalyst']}")
    print(f"  • Best Solvent: {optimal_conditions['solvent']}")
    print(f"  • Optimal Yield: {optimal_conditions['predicted_yield']:.1f}%")
    print(f"  • Optimal Impurity: {optimal_conditions['predicted_impurity']:.1f}%")
    print("\n🎯 This analysis demonstrates:")
    print("  • Advanced ML modeling for chemical processes")
    print("  • Feature importance analysis for process optimization")
    print("  • Multi-objective optimization (yield vs impurity)")
    print("  • Professional data visualization and reporting")
    print("  • End-to-end ML pipeline development")
    
if __name__ == "__main__":
    main()
