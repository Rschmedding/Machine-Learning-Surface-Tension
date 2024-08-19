
!pip install xgboost rdkit
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from xgboost import XGBRegressor

# Function to check file existence
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

# Function to read input data from the specified file
def read_input_data(filename):
    df = pd.read_csv(filename, encoding='ISO-8859-1')
    required_columns = ['SMILES', 'TempC']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain columns: {', '.join(required_columns)}")
    df['TempC'] = pd.to_numeric(df['TempC'], errors='coerce')
    return df.dropna(subset=required_columns)

# Function to calculate molecular properties
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

        return [molar_weight, oc_ratio, hc_ratio, nc_ratio, sc_ratio, pc_ratio,
                cl_c_ratio, f_c_ratio, i_c_ratio, br_c_ratio]
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return [np.nan] * 10

# Function to convert SMILES codes to MACCS keys
def smiles_to_maccs(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        maccs = MACCSkeys.GenMACCSKeys(mol)
        return np.array(maccs)
    except:
        return None

# Function to convert SMILES codes to Morgan fingerprints
def smiles_to_morgan(smiles, radius=2, nBits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return np.array(morgan)
    except:
        return None

# Load pre-trained models
Morgan_best_model_fp = "./Saved_Models/20240808_XGBoost_Morgan.json"
MACCS_best_model_fp = "./Saved_Models/20240808_XGBoost_MACSS.json"
Simplified_best_model_fp = "./Saved_Models/20240812_XGBoost_Simplified.json"

Morgan_best_model = XGBRegressor()
Morgan_best_model.load_model(Morgan_best_model_fp)

MACCS_best_model = XGBRegressor()
MACCS_best_model.load_model(MACCS_best_model_fp)

Simplified_best_model = XGBRegressor()
Simplified_best_model.load_model(Simplified_best_model_fp)

# Main script
def main():
    # Prompt the user for the file number
    file_number = input("Enter the file number: ")
    input_file_path = f"./Model_Inputs/Input_{file_number}.csv"
    
    # Check if the file exists
    check_file_exists(input_file_path)
    
    # Read the input data
    df = read_input_data(input_file_path)
    
    # Process and predict using MACCS model
    df['MACCS_Keys'] = df.apply(lambda row: smiles_to_maccs(row['SMILES']), axis=1)
    df = df.dropna(subset=["MACCS_Keys"]).reset_index(drop=True)
    X_temp = df[['TempC']].values + 273  # ensure temp is in kelvin
    X_maccs = np.array(list(df['MACCS_Keys']))
    X_scaled = np.concatenate((X_temp, X_maccs), axis=1)
    df["Sigma_MACCS"] = MACCS_best_model.predict(X_scaled)
    
    # Process and predict using Morgan model
    df['Morgan'] = df.apply(lambda row: smiles_to_morgan(row['SMILES']), axis=1)
    df = df.dropna(subset=["Morgan"]).reset_index(drop=True)
    X_morgan = np.array(list(df['Morgan']))
    X_scaled = np.concatenate((X_temp, X_morgan), axis=1)
    df["Sigma_Morgan"] = Morgan_best_model.predict(X_scaled)
    
    # Process and predict using Simplified model
    df['MolecularProperties'] = df['SMILES'].apply(calculate_molecular_properties)
    properties_df = pd.DataFrame(df['MolecularProperties'].tolist(), columns=[
        'MolarWeight', 'OC_Ratio', 'HC_Ratio', 'NC_Ratio', 'SC_Ratio', 'PC_Ratio',
        'Cl_C_Ratio', 'F_C_Ratio', 'I_C_Ratio', 'Br_C_Ratio'
    ])
    df = pd.concat([df, properties_df], axis=1).dropna().reset_index(drop=True)
    df['Temperature'] = df['TempC'] + 273
    X_simplified = df[['Temperature', 'MolarWeight', 'OC_Ratio', 'HC_Ratio', 'NC_Ratio', 'SC_Ratio', 'PC_Ratio',
        'Cl_C_Ratio', 'F_C_Ratio', 'I_C_Ratio', 'Br_C_Ratio']].values
    df["Sigma_Simplified"] = Simplified_best_model.predict(X_simplified)

    # Save the output to the Model_Outputs folder
    output_file_path = f"./Model_Outputs/Output_{file_number}.csv"
    df_out = df[["SMILES", "TempC", "Sigma_MACCS", "Sigma_Morgan", "Sigma_Simplified"]]
    df_out.to_csv(output_file_path, index=False)
    print(f"Predictions saved to {output_file_path}")

if __name__ == "__main__":
    main()
