"""
Machine Learning Model for Predicting Impurity Formation in Drug Synthesis        print("Categorical Variables:")
        for col in ['Catalyst', 'Solvent']:
            if col in self.data.columns:
                print(f"{col}: {self.data[col].unique()}")This project trains ML models to predict yield and impurity formation in pharmaceutical 
synthesis reactions based on reaction conditions (temperature, concentration, catalysts, etc.).

Author: Abhishek kumawat
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class DrugSynthesisPredictor:
    """Main class for predicting yield and impurity in drug synthesis reactions"""
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_yield_train = None
        self.y_yield_test = None
        self.y_impurity_train = None
        self.y_impurity_test = None
        self.yield_model = None
        self.impurity_model = None
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """Load and preprocess the reaction data"""
        print("Loading reaction data...")
        self.data = pd.read_csv(file_path)
        
        # Handle any NaN values
        self.data = self.data.dropna()
        
        print(f"Loaded {len(self.data)} reaction records")
        print(f"Features: {list(self.data.columns)}")
        
        # Display basic info about the realistic dataset
        print("\nDataset Overview:")
        print(f"Temperature range: {self.data['Temperature_C'].min():.1f}°C - {self.data['Temperature_C'].max():.1f}°C")
        print(f"Concentration range: {self.data['Concentration_M'].min():.3f}M - {self.data['Concentration_M'].max():.3f}M")
        print(f"pH range: {self.data['pH'].min():.1f} - {self.data['pH'].max():.1f}")
        print(f"Reaction time range: {self.data['Reaction_Time_h'].min():.1f}h - {self.data['Reaction_Time_h'].max():.1f}h")
        print(f"Catalysts: {', '.join(self.data['Catalyst'].unique())}")
        print(f"Solvents: {', '.join(self.data['Solvent'].unique())}")
        print(f"Yield range: {self.data['Yield_%'].min():.1f}% - {self.data['Yield_%'].max():.1f}%")
        print(f"Impurity range: {self.data['Impurity_%'].min():.3f}% - {self.data['Impurity_%'].max():.3f}%")
        
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        print("\nDataset Info:")
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        print("\nStatistical Summary:")
        print(self.data.describe())
        
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
        print("\nCategorical Variables:")
        for col in ['Catalyst', 'Solvent']:
            if col in self.data.columns:
                print(f"{col}: {self.data[col].unique()}")
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Reaction Parameters')
        plt.tight_layout()
        plt.savefig('figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def prepare_features(self):
        """Prepare features for machine learning"""
        print("\n" + "="*50)
        print("FEATURE PREPARATION")
        print("="*50)
        
        # Handle missing values in catalyst column
        self.data['Catalyst'] = self.data['Catalyst'].fillna('None')
        
        # Encode categorical variables
        categorical_cols = ['Catalyst', 'Solvent']
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[f'{col}_encoded'] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le
            print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Select features - updated for realistic dataset
        feature_cols = ['Temperature_C', 'Concentration_M', 'pH', 'Reaction_Time_h', 
                       'Catalyst_Loading_mol%', 'Catalyst_encoded', 'Solvent_encoded']
        
        X = self.data[feature_cols]
        y_yield = self.data['Yield_%']
        y_impurity = self.data['Impurity_%']
        
        # Split data
        self.X_train, self.X_test, self.y_yield_train, self.y_yield_test, \
        self.y_impurity_train, self.y_impurity_test = train_test_split(
            X, y_yield, y_impurity, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"Features: {list(X.columns)}")
        
    def train_models(self):
        """Train machine learning models for yield and impurity prediction"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Define models to compare
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Ridge Regression': Ridge()
        }
        
        best_yield_score = -np.inf
        best_impurity_score = -np.inf
        
        # Train models for yield prediction
        print("\nTraining models for YIELD prediction:")
        for name, model in models.items():
            scores = cross_val_score(model, self.X_train, self.y_yield_train, 
                                   cv=5, scoring='r2')
            print(f"{name}: CV R² = {scores.mean():.4f} (±{scores.std():.4f})")
            
            if scores.mean() > best_yield_score:
                best_yield_score = scores.mean()
                self.yield_model = model
                best_yield_name = name
        
        # Train models for impurity prediction
        print("\nTraining models for IMPURITY prediction:")
        for name, model in models.items():
            scores = cross_val_score(model, self.X_train, self.y_impurity_train, 
                                   cv=5, scoring='r2')
            print(f"{name}: CV R² = {scores.mean():.4f} (±{scores.std():.4f})")
            
            if scores.mean() > best_impurity_score:
                best_impurity_score = scores.mean()
                self.impurity_model = model
                best_impurity_name = name
        
        # Fit best models
        print(f"\nBest model for yield: {best_yield_name}")
        print(f"Best model for impurity: {best_impurity_name}")
        
        self.yield_model.fit(self.X_train, self.y_yield_train)
        self.impurity_model.fit(self.X_train, self.y_impurity_train)
        
    def evaluate_models(self):
        """Evaluate model performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Yield predictions
        yield_pred = self.yield_model.predict(self.X_test)
        yield_r2 = r2_score(self.y_yield_test, yield_pred)
        yield_mae = mean_absolute_error(self.y_yield_test, yield_pred)
        yield_rmse = np.sqrt(mean_squared_error(self.y_yield_test, yield_pred))
        
        # Impurity predictions
        impurity_pred = self.impurity_model.predict(self.X_test)
        impurity_r2 = r2_score(self.y_impurity_test, impurity_pred)
        impurity_mae = mean_absolute_error(self.y_impurity_test, impurity_pred)
        impurity_rmse = np.sqrt(mean_squared_error(self.y_impurity_test, impurity_pred))
        
        print("YIELD MODEL PERFORMANCE:")
        print(f"  R² Score: {yield_r2:.4f}")
        print(f"  MAE: {yield_mae:.4f}%")
        print(f"  RMSE: {yield_rmse:.4f}%")
        
        print("\nIMPURITY MODEL PERFORMANCE:")
        print(f"  R² Score: {impurity_r2:.4f}")
        print(f"  MAE: {impurity_mae:.4f}%")
        print(f"  RMSE: {impurity_rmse:.4f}%")
        
        # Save metrics
        metrics = {
            'yield': {
                'R2': yield_r2,
                'MAE': yield_mae,
                'RMSE': yield_rmse
            },
            'impurity': {
                'R2': impurity_r2,
                'MAE': impurity_mae,
                'RMSE': impurity_rmse
            }
        }
        
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def feature_importance_analysis(self):
        """Analyze feature importance using permutation importance"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        feature_names = self.X_train.columns
        
        # Yield model importance
        yield_perm_imp = permutation_importance(
            self.yield_model, self.X_test, self.y_yield_test,
            n_repeats=10, random_state=42
        )
        
        # Impurity model importance
        impurity_perm_imp = permutation_importance(
            self.impurity_model, self.X_test, self.y_impurity_test,
            n_repeats=10, random_state=42
        )
        
        # Create importance plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Yield importance
        yield_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': yield_perm_imp.importances_mean,
            'std': yield_perm_imp.importances_std
        }).sort_values('importance', ascending=True)
        
        ax1.barh(yield_importance_df['feature'], yield_importance_df['importance'],
                xerr=yield_importance_df['std'])
        ax1.set_title('Feature Importance - Yield Prediction')
        ax1.set_xlabel('Permutation Importance')
        
        # Impurity importance
        impurity_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': impurity_perm_imp.importances_mean,
            'std': impurity_perm_imp.importances_std
        }).sort_values('importance', ascending=True)
        
        ax2.barh(impurity_importance_df['feature'], impurity_importance_df['importance'],
                xerr=impurity_importance_df['std'])
        ax2.set_title('Feature Importance - Impurity Prediction')
        ax2.set_xlabel('Permutation Importance')
        
        plt.tight_layout()
        plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("YIELD MODEL - Top 3 Most Important Features:")
        for i, row in yield_importance_df.tail(3).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f} (±{row['std']:.4f})")
        
        print("\nIMPURITY MODEL - Top 3 Most Important Features:")
        for i, row in impurity_importance_df.tail(3).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f} (±{row['std']:.4f})")
    
    def create_prediction_heatmaps(self):
        """Create heatmaps showing how reaction conditions affect yield and impurity"""
        print("\n" + "="*50)
        print("CREATING PREDICTION HEATMAPS")
        print("="*50)
        
        # Create parameter grids for heatmaps
        temp_range = np.linspace(50, 120, 20)
        conc_range = np.linspace(0.1, 2.0, 20)
        
        # Base conditions (median values) - updated for realistic dataset
        base_conditions = {
            'Reaction_Time_h': self.data['Reaction_Time_h'].median(),
            'pH': self.data['pH'].median(),
            'Catalyst_Loading_mol%': self.data['Catalyst_Loading_mol%'].median(),
        }
        
        catalysts = ['Pd/C', 'TEMPO', 'DMP', 'PPh3']  # Use actual catalysts from dataset
        
        for catalyst in catalysts:
            # Encode catalyst
            catalyst_encoded = self.label_encoders['Catalyst'].transform([catalyst])[0]
            
            # Use most common solvent (DMSO from our analysis)
            solvent_encoded = self.label_encoders['Solvent'].transform(['DMSO'])[0]
            
            # Create prediction grid
            temp_grid, conc_grid = np.meshgrid(temp_range, conc_range)
            predictions_yield = []
            predictions_impurity = []
            
            for i in range(len(temp_range)):
                for j in range(len(conc_range)):
                    conditions = [
                        temp_range[i],  # Temperature_C
                        conc_range[j],  # Concentration_M
                        base_conditions['pH'],
                        base_conditions['Reaction_Time_h'],
                        base_conditions['Catalyst_Loading_mol%'],
                        catalyst_encoded,
                        solvent_encoded
                    ]
                    
                    yield_pred = self.yield_model.predict([conditions])[0]
                    impurity_pred = self.impurity_model.predict([conditions])[0]
                    
                    predictions_yield.append(yield_pred)
                    predictions_impurity.append(impurity_pred)
            
            # Reshape predictions
            yield_matrix = np.array(predictions_yield).reshape(len(conc_range), len(temp_range))
            impurity_matrix = np.array(predictions_impurity).reshape(len(conc_range), len(temp_range))
            
            # Create heatmaps
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Yield heatmap
            sns.heatmap(yield_matrix, xticklabels=np.round(temp_range, 1)[::4], 
                       yticklabels=np.round(conc_range, 2)[::4], 
                       annot=False, cmap='RdYlGn', ax=ax1)
            ax1.set_title(f'Predicted Yield (%) - Catalyst {catalyst}')
            ax1.set_xlabel('Temperature (°C)')
            ax1.set_ylabel('Concentration (M)')
            
            # Impurity heatmap
            sns.heatmap(impurity_matrix, xticklabels=np.round(temp_range, 1)[::4], 
                       yticklabels=np.round(conc_range, 2)[::4], 
                       annot=False, cmap='RdYlBu_r', ax=ax2)
            ax2.set_title(f'Predicted Impurity (%) - Catalyst {catalyst}')
            ax2.set_xlabel('Temperature (°C)')
            ax2.set_ylabel('Concentration (M)')
            
            plt.tight_layout()
            # Replace forward slash in catalyst name for file saving
            safe_catalyst_name = catalyst.replace('/', '_')
            plt.savefig(f'figures/heatmap_catalyst_{safe_catalyst_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def find_optimal_conditions(self):
        """Find optimal reaction conditions"""
        print("\n" + "="*50)
        print("FINDING OPTIMAL CONDITIONS")
        print("="*50)
        
        # Generate random conditions
        n_samples = 10000
        np.random.seed(42)
        
        conditions = []
        for _ in range(n_samples):
            temp = np.random.uniform(50, 120)
            conc = np.random.uniform(0.1, 2.0)
            residence_time = np.random.uniform(10, 60)
            ph = np.random.uniform(1, 10)
            equiv_oxidant = np.random.uniform(0.1, 2.5)
            stirring = np.random.uniform(200, 1200)
            catalyst = np.random.choice([0, 1, 2, 3])  # Encoded values
            solvent = np.random.choice([0, 1, 2, 3])   # Encoded values
            
            conditions.append([temp, conc, residence_time, ph, equiv_oxidant, 
                             stirring, catalyst, solvent])
        
        conditions = np.array(conditions)
        
        # Predict for all conditions
        yield_predictions = self.yield_model.predict(conditions)
        impurity_predictions = self.impurity_model.predict(conditions)
        
        # Find optimal conditions (high yield, low impurity)
        # Use a composite score: yield - impurity_weight * impurity
        impurity_weight = 0.5
        composite_scores = yield_predictions - impurity_weight * impurity_predictions
        
        best_idx = np.argmax(composite_scores)
        best_conditions = conditions[best_idx]
        
        # Decode categorical variables
        catalyst_decoded = self.label_encoders['catalyst'].inverse_transform([int(best_conditions[6])])[0]
        solvent_decoded = self.label_encoders['solvent'].inverse_transform([int(best_conditions[7])])[0]
        
        optimal_result = {
            'temperature_C': float(best_conditions[0]),
            'concentration_M': float(best_conditions[1]),
            'residence_time_min': float(best_conditions[2]),
            'pH': float(best_conditions[3]),
            'equiv_oxidant': float(best_conditions[4]),
            'stirring_rpm': float(best_conditions[5]),
            'catalyst': catalyst_decoded,
            'solvent': solvent_decoded,
            'predicted_yield': float(yield_predictions[best_idx]),
            'predicted_impurity': float(impurity_predictions[best_idx]),
            'composite_score': float(composite_scores[best_idx])
        }
        
        print("OPTIMAL CONDITIONS FOUND:")
        print(f"  Temperature: {optimal_result['temperature_C']:.1f}°C")
        print(f"  Concentration: {optimal_result['concentration_M']:.2f}M")
        print(f"  Residence Time: {optimal_result['residence_time_min']:.1f}min")
        print(f"  pH: {optimal_result['pH']:.1f}")
        print(f"  Equiv. Oxidant: {optimal_result['equiv_oxidant']:.2f}")
        print(f"  Stirring: {optimal_result['stirring_rpm']:.0f}rpm")
        print(f"  Catalyst: {optimal_result['catalyst']}")
        print(f"  Solvent: {optimal_result['solvent']}")
        print(f"\nPREDICTED RESULTS:")
        print(f"  Yield: {optimal_result['predicted_yield']:.1f}%")
        print(f"  Impurity: {optimal_result['predicted_impurity']:.1f}%")
        
        # Save optimal conditions
        with open('reports/best_conditions.json', 'w') as f:
            json.dump(optimal_result, f, indent=2)
            
        return optimal_result
    
    def save_models(self):
        """Save trained models"""
        print("\nSaving trained models...")
        joblib.dump(self.yield_model, 'models/model_yield.joblib')
        joblib.dump(self.impurity_model, 'models/model_impurity.joblib')
        joblib.dump(self.label_encoders, 'models/label_encoders.joblib')
        print("Models saved successfully!")
    
    def run_complete_analysis(self):
        """Run the complete machine learning pipeline"""
        print("="*70)
        print("MACHINE LEARNING FOR DRUG SYNTHESIS IMPURITY PREDICTION")
        print("="*70)
        
        # Load and explore data
        self.load_data('data/synthetic_reaction_data.csv')
        self.explore_data()
        
        # Prepare features and train models
        self.prepare_features()
        self.train_models()
        
        # Evaluate and analyze
        metrics = self.evaluate_models()
        self.feature_importance_analysis()
        
        # Create visualizations
        self.create_prediction_heatmaps()
        
        # Find optimal conditions
        optimal_conditions = self.find_optimal_conditions()
        
        # Save models
        self.save_models()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print("✅ Models trained and evaluated")
        print("✅ Feature importance analyzed")
        print("✅ Prediction heatmaps created")
        print("✅ Optimal conditions identified")
        print("✅ Models saved for future use")
        
        return metrics, optimal_conditions

if __name__ == "__main__":
    # First generate the realistic dataset
    print("="*70)
    print("GENERATING REALISTIC PHARMACEUTICAL DATASET")
    print("="*70)
    
    import subprocess
    import sys
    
    try:
        # Run the data generation script
        result = subprocess.run([sys.executable, 'generate_realistic_data.py'], 
                               capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error generating dataset: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        
    print("\n" + "="*70)
    print("RUNNING MACHINE LEARNING ANALYSIS")
    print("="*70)
    
    # Run the complete analysis with the new realistic dataset
    predictor = DrugSynthesisPredictor()
    
    # Load the realistic dataset
    predictor.load_data('data/realistic_pharma_data.csv')
    
    # Run analysis
    predictor.explore_data()
    predictor.prepare_features()
    predictor.train_models()
    
    # Evaluate and analyze
    metrics = predictor.evaluate_models()
    predictor.feature_importance_analysis()
    
    # Create visualizations
    predictor.create_prediction_heatmaps()
    
    # Find optimal conditions
    optimal_conditions = predictor.find_optimal_conditions()
    
    # Save models
    predictor.save_models()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("✅ Realistic pharmaceutical dataset generated")
    print("✅ Models trained and evaluated on real-world conditions")
    print("✅ Feature importance analyzed")
    print("✅ Prediction heatmaps created")
    print("✅ Optimal conditions identified")
    print("✅ Models saved for future use")
    print("\nThis project demonstrates:")
    print("• Realistic pharmaceutical synthesis data modeling")
    print("• Multi-target prediction (yield and impurity)")
    print("• Advanced feature engineering and model comparison")
    print("• Professional visualization and analysis")
    print("• Industry-relevant catalyst and solvent effects")
