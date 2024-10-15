# Necessary imports and installations
# !pip install optuna xgboost rdkit pyswarm

from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from xgboost import XGBRegressor
import joblib
import pickle
import matplotlib.pyplot as plt

# Set working directory
os.chdir("/home/jovyan/ML_Project")

# Read input data
def read_input_data(filename):
    df = pd.read_csv(filename, encoding='ISO-8859-1')
    required_columns = ['SMILES', 'TempC', 'Sigma']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain columns: {', '.join(required_columns)}")
    df['TempC'] = pd.to_numeric(df['TempC'], errors='coerce')
    df['Sigma'] = pd.to_numeric(df['Sigma'], errors='coerce')
    return df.dropna(subset=required_columns)

# Calculate molecular properties
def calculate_molecular_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        molar_weight = Descriptors.MolWt(mol)
        num_carbon = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        num_oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
        num_hydrogen = sum(atom.GetTotalNumHs(includeNeighbors=True) for atom in mol.GetAtoms())
        num_nitrogen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
        num_sulfur = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S')
        num_phosphorus = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'P')
        num_chlorine = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Cl')
        num_fluorine = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
        num_iodine = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'I')
        num_bromine = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Br')

        oc_ratio = num_oxygen / num_carbon if num_carbon else 0
        hc_ratio = num_hydrogen / num_carbon if num_carbon else 0
        nc_ratio = num_nitrogen / num_carbon if num_carbon else 0
        sc_ratio = num_sulfur / num_carbon if num_carbon else 0
        pc_ratio = num_phosphorus / num_carbon if num_carbon else 0
        cl_c_ratio = num_chlorine / num_carbon if num_carbon else 0
        f_c_ratio = num_fluorine / num_carbon if num_carbon else 0
        i_c_ratio = num_iodine / num_carbon if num_carbon else 0
        br_c_ratio = num_bromine / num_carbon if num_carbon else 0

        return [molar_weight, oc_ratio, hc_ratio, nc_ratio, sc_ratio, pc_ratio, cl_c_ratio, f_c_ratio, i_c_ratio, br_c_ratio]
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return [np.nan] * 11

# SMILES to MACCS
def smiles_to_maccs(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    except:
        return None

# SMILES to Morgan fingerprints
def smiles_to_morgan(smiles, radius=2, nBits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))
    except:
        return None

# Optimize Gradient Boosting Machine (XGBoost)
def optimize_gbm(X_train, y_train, X_val, y_val, n_trials=100):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'eta': trial.suggest_float('eta', 1e-3, 1.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-5, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),
            'lambda': trial.suggest_float('lambda', 1e-5, 1.0, log=True),
            'random_state': 42,
            'tree_method': 'hist',
            'objective': 'reg:squarederror'
        }
        model = XGBRegressor(**params)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        mse_scores = [mean_squared_error(y_val_fold, model.fit(X_train_fold, y_train_fold).predict(X_val_fold))
                      for train_index, val_index in kf.split(X_train, y_train)
                      for X_train_fold, X_val_fold, y_train_fold, y_val_fold in 
                      [(X_train[train_index], X_train[val_index], y_train[train_index], y_train[val_index])]]
        return np.mean(mse_scores)

    study = optuna.create_study(direction='minimize', pruner=SuccessiveHalvingPruner())
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params

    model_gbm = XGBRegressor(**best_params, random_state=42, objective='reg:squarederror')
    model_gbm.fit(X_train, y_train)
    preds_val = model_gbm.predict(X_val)
    return model_gbm, mean_squared_error(y_val, preds_val), r2_score(y_val, preds_val), best_params

# Optimize Random Forest
def optimize_rf(X_train, y_train, X_val, y_val, n_trials=50):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42
        }
        model = RandomForestRegressor(**params)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        mse_scores = [mean_squared_error(y_val_fold, model.fit(X_train_fold, y_train_fold).predict(X_val_fold))
                      for train_index, val_index in kf.split(X_train, y_train)
                      for X_train_fold, X_val_fold, y_train_fold, y_val_fold in 
                      [(X_train[train_index], X_train[val_index], y_train[train_index], y_train[val_index])]]
        return np.mean(mse_scores)

    study = optuna.create_study(direction='minimize', pruner=SuccessiveHalvingPruner())
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params

    model_rf = RandomForestRegressor(**best_params, random_state=42)
    model_rf.fit(X_train, y_train)
    preds_val = model_rf.predict(X_val)
    return model_rf, mean_squared_error(y_val, preds_val), r2_score(y_val, preds_val), best_params

# Optimize Decision Tree
def optimize_dt(X_train, y_train, X_val, y_val, n_trials=50):
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'random_state': 42
        }
        model = DecisionTreeRegressor(**params)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        mse_scores = [mean_squared_error(y_val_fold, model.fit(X_train_fold, y_train_fold).predict(X_val_fold))
                      for train_index, val_index in kf.split(X_train, y_train)
                      for X_train_fold, X_val_fold, y_train_fold, y_val_fold in 
                      [(X_train[train_index], X_train[val_index], y_train[train_index], y_train[val_index])]]
        return np.mean(mse_scores)

    study = optuna.create_study(direction='minimize', pruner=SuccessiveHalvingPruner())
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params

    model_dt = DecisionTreeRegressor(**best_params, random_state=42)
    model_dt.fit(X_train, y_train)
    preds_val = model_dt.predict(X_val)
    return model_dt, mean_squared_error(y_val, preds_val), r2_score(y_val, preds_val), best_params

# Optimize KNN
# Optimize K-Nearest Neighbors (KNN)
def optimize_knn(X_train, y_train, X_val, y_val, n_trials=50):
    def objective(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 50),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
            'leaf_size': trial.suggest_int('leaf_size', 10, 100),
            'p': trial.suggest_int('p', 1, 5)  # 1: Manhattan, 2: Euclidean
        }

        model = KNeighborsRegressor(**params)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        mse_scores = [mean_squared_error(y_val_fold, model.fit(X_train_fold, y_train_fold).predict(X_val_fold))
                      for train_index, val_index in kf.split(X_train, y_train)
                      for X_train_fold, X_val_fold, y_train_fold, y_val_fold in 
                      [(X_train[train_index], X_train[val_index], y_train[train_index], y_train[val_index])]]
        return np.mean(mse_scores)

    study = optuna.create_study(direction='minimize', pruner=SuccessiveHalvingPruner())
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params

    model_knn = KNeighborsRegressor(**best_params)
    model_knn.fit(X_train, y_train)
    preds_val = model_knn.predict(X_val)
    return model_knn, mean_squared_error(y_val, preds_val), r2_score(y_val, preds_val), best_params

input_file = "./Model_Inputs/Training_data.csv"
df = read_input_data(input_file)
input_file = "./Model_Inputs/Validation_data.csv"
df_val = read_input_data(input_file)
                        
# Apply the function and drop rows where conversion failed
df['Morgan_FP'] = df.apply(lambda row: smiles_to_morgan(row['SMILES']), axis=1)
df_val['Morgan_FP'] = df_val.apply(lambda row: smiles_to_morgan(row['SMILES']), axis=1)

# Drop rows with missing values
df = df.dropna(subset=["SMILES", "Sigma",'Morgan_FP']).reset_index(drop=True)
df_val = df_val.dropna(subset=["SMILES", "Sigma",'Morgan_FP']).reset_index(drop=True)

# Define input features and target variable
X_temp = df[['TempC']].values #+ 273  # ensure temp is in kelvin
X_morgan = np.array(list(df['Morgan_FP']))
X = np.concatenate((X_temp, X_morgan), axis=1)
y = df['Sigma'].values

# Standardize the input features
X_scaled =X#scaler.fit_transform(X)

# Define input features and target variable
X_temp_val = df_val[['TempC']].values #+ 273  # ensure temp is in kelvin
X_morgan_val = np.array(list(df_val['Morgan_FP']))
X_val = np.concatenate((X_temp_val, X_morgan_val), axis=1)
y_val = df_val['Sigma'].values


# Model optimization and evaluation
models = {
    'KNN': (optimize_knn, [X_simplified, y]),
    'Decision Tree': (optimize_dt, [X_simplified, y]),
    'Random Forest': (optimize_rf, [X_simplified, y]),
	'XGBoost':(optimize_gbm, [X_simplified, y])
}

# Define plot settings for consistent styling
plot_colors = {
    'point': '#9467bd',  # dark purple (colorblind-friendly)
    'line': '#ff7f0e'    # orange (colorblind-friendly)
}

# Iterate through models, optimize, and generate performance metrics
for model_name, (optimizer_func, args) in models.items():
    # Perform optimization and validation
    optimizer_result = optimizer_func(*args, X_val, y_val)
    model, mse_val, r2_val, best_params = optimizer_result
    print(f"{model_name}: Validation MSE: {mse_val}, Validation R2: {r2_val}")

    # Save the model
    model_filename = f"./Saved_Models/{datetime.now().strftime('%Y%m%d')}_{model_name.replace(' ', '_')}_Simplified.pkl"
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)

    # Generate validation plot
    preds = model.predict(X_simplified_val)
    plt.scatter(y_val, preds, color=plot_colors['point'], edgecolor='black', alpha=0.7)
    plt.plot([0, 100], [0, 100], color=plot_colors['line'], linestyle='--', linewidth=1)
    plt.xlabel(r'Reported $\sigma_i^{\circ}$ $[\mathrm{mJ~m^{-2}}]$', fontsize=24)
    plt.ylabel(r'Predicted $\sigma_i^{\circ}$ $[\mathrm{mJ~m^{-2}}]$', fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.xlim(0.9 * min(y_val), 1.1 * max(y_val))
    plt.ylim(0.9 * min(y_val), 1.1 * max(y_val))
    plot_filename = f"./Figures/{datetime.now().strftime('%Y%m%d')}_{model_name.replace(' ', '_')}_Validation_Plot_Simplified_Optuna.pdf"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Validation plot saved: {plot_filename}")

params =  {'n_estimators': 1429, 
           'max_depth': 7, 
           'min_child_weight': 7, 
           'subsample': 0.9974278880962442, 
           'colsample_bytree': 0.7157348836579956, 
           'eta': 0.3696634733118418, 
           'gamma': 0.04062302411740389, 
           'alpha': 0.0005594394731173785, 
           'lambda': 0.003022035839898325,
            'random_state': 42,
            'monotone_constraints': (-1,) + (0,) * 2048,
            'tree_method': 'hist',
            'objective': 'reg:squarederror'}

model_gbm = XGBRegressor(**params)
model_gbm.fit(X, y)

# Predict and evaluate the XGBoost model
preds_val = model_gbm.predict(X_val)
mse_val = mean_squared_error(y_val, preds_val)
r2_val = r2_score(y_val, preds_val)
print(f"XGBoost - MSE: {mse_val}, R^2: {r2_val}")

# Generate XGBoost validation plot
plt.scatter(y_val, preds_val, edgecolor='black', alpha=0.7)
plt.plot([0, 100], [0, 100], linestyle='--', linewidth=1)
plt.xlabel(r'Reported $\sigma_i^{\circ}$ $[\mathrm{mJ~m^{-2}}]$', fontsize=24)
plt.ylabel(r'Predicted $\sigma_i^{\circ}$ $[\mathrm{mJ~m^{-2}}]$', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.xlim(0.9 * min(y_val), 1.1 * max(y_val))
plt.ylim(0.9 * min(y_val), 1.1 * max(y_val))
plot_filename = f"./Figures/{datetime.now().strftime('%Y%m%d')}_XGBoost_Validation_Plot_Simplified_Optuna.pdf"
plt.savefig(plot_filename)
plt.close()
print(f"XGBoost validation plot saved: {plot_filename}")

# Save the XGBoost model
model_filename = f"./Saved_Models/{datetime.now().strftime('%Y%m%d')}_XGBoost_Simplified.bin"
model_gbm.save_model(model_filename)

