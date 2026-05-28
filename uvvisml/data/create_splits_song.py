import os
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from scaffold_splits import scaffold_split

def remove_invalid_smiles(df, smiles_col='smiles'):
    """Remove rows with invalid SMILES."""
    mask = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)
    return df.loc[mask].reset_index(drop=True)


def remove_nan_targets(df, target_col):
    """Remove rows where target is NaN."""
    return df.dropna(subset=[target_col]).reset_index(drop=True)


def basic_clean(df, smiles_col, target_col):
    """Standard cleaning used for all datasets."""
    df = df.copy()

    # normalize smiles column name
    if smiles_col != 'smiles':
        df = df.rename(columns={smiles_col: 'smiles'})

    # remove NaNs
    df = remove_nan_targets(df, target_col)

    # remove invalid molecules
    df = remove_invalid_smiles(df)

    return df.reset_index(drop=True)

def data_split_and_write(X, feature_names=None, target_names=['peakwavs_max'], solvation=False, split_type='scaffold',
                         scale_targets=False, write_files=False, random_seed=0):
    """Writes train, val, test CSV files for Chemprop with a given dataset.

    Parameters
    ----------
    X : pandas DataFrame
        DataFrame to be split (has columns: 'smiles', target_names, and feature_names
        (and 'solvent' if solvation=True))
    feature_names : list of str or None
        Names of feature columns in X to be added to feature files (default is None)
    target_names : list of str
        Names of target columns to be printed to files (default is ['peakwavs_max'])
    solvation : bool
        Specify whether to include solvents in target file (default is False)
    split_type : str
        which type of splitting to use ('scaffold', 'group_by_smiles', or 'random')
    scale_targets : bool
        whether to scale targets to have mean 0 and standard deviation 1 (default is False)
    write_files : bool
        whether to write the resulting splits to files
    random_seed : int or None
        number to provide for the seed / random_state arguments to make splits reproducible
        (use None if doing multiple splits for cross validation)

    """

    if split_type=='scaffold':
        X_train, X_val, X_test = scaffold_split(X, sizes=(0.8, 0.1, 0.1), balanced=True, seed=random_seed)
    elif split_type=='group_by_smiles': # Randomly split into train, val, and test sets such that no SMILES is in multiple sets
        gss1 = GroupShuffleSplit(n_splits=2, train_size=0.8, random_state=random_seed)
        train_idx, temp_idx = list(gss1.split(X, groups=X['smiles']))[0]
        X_train, X_temp = X.iloc[train_idx,:], X.iloc[temp_idx,:]
        gss2 = GroupShuffleSplit(n_splits=2, train_size=0.5, random_state=random_seed)
        val_idx, test_idx = list(gss2.split(X_temp, groups=X_temp['smiles']))[0]
        X_val, X_test = X_temp.iloc[val_idx, :], X_temp.iloc[test_idx, :]
    elif split_type=='random': # Randomly split into train, val, and test sets
        X_train = X.sample(frac=0.8, random_state=random_seed)
        X_temp = X.drop(X_train.index)
        X_val = X_temp.sample(frac=0.5, random_state=random_seed)
        X_test = X_temp.drop(X_val.index)

    if scale_targets:
        scaler = StandardScaler()
        scaler.fit(X_train[['target_names']])
        X_train[['target_names']] = scaler.transform(X_train[['target_names']])
        X_val[['target_names']] = scaler.transform(X_val[['target_names']])
        X_test[['target_names']] = scaler.transform(X_test[['target_names']])

    if write_files:
        # Name files
        train_target_file = 'smiles_target_train.csv'
        val_target_file = 'smiles_target_val.csv'
        test_target_file = 'smiles_target_test.csv'
        train_features_file = 'features_train.csv'
        val_features_file = 'features_val.csv'
        test_features_file = 'features_test.csv'

        # Write splits to CSVs
        if solvation:
            X_train[['smiles','solvent']+target_names].to_csv(train_target_file, index=False)
            X_val[['smiles','solvent']+target_names].to_csv(val_target_file, index=False)
            X_test[['smiles','solvent']+target_names].to_csv(test_target_file, index=False)
        else:
            X_train[['smiles']+target_names].to_csv(train_target_file, index=False)
            X_val[['smiles']+target_names].to_csv(val_target_file, index=False)
            X_test[['smiles']+target_names].to_csv(test_target_file, index=False)
        if feature_names:
            X_train[feature_names].to_csv(train_features_file, index=False)
            X_val[feature_names].to_csv(val_features_file, index=False)
            X_test[feature_names].to_csv(test_features_file, index=False)

    return X_train, X_val, X_test

def handle_duplicates(df, target_col, cutoff=5):

    cols = [x for x in df.columns if x not in ['smiles','solvent',target_col,'source']]

    agg_dict = {target_col:['mean','std']}

    if 'source' in df.columns:
        agg_dict['source'] = lambda x: 'multiple' if len(x) > 1 else x.iloc[0]

    for col in cols:
        agg_dict[col] = 'mean'

    # aggregate
    if 'solvent' in df.columns:
        group_cols = ['smiles','solvent']
    else:
        group_cols = ['smiles']

    df = df.groupby(group_cols).agg(agg_dict).reset_index()

    # remove high variance duplicates
    high_std_idx = df[df[target_col]['std'] > cutoff].index
    df.drop(index=high_std_idx, inplace=True)

    df.drop(columns='std', level=1, inplace=True)
    df.columns = df.columns.get_level_values(0)

    return df

# Set initial directory
DATA_DIR = os.getcwd()

# Load Song datasets
fluodb_path = r"uvvisml\uvvisml\data\original\song\00_FluoDB.csv"
consolidation_path = r"uvvisml\uvvisml\data\original\song\Dataset_Consolidation.csv"

fluodb_df = pd.read_csv(fluodb_path)
consolidation_df = pd.read_csv(consolidation_path)


# -----------------------
# Clean FluoDB dataset
# -----------------------

consolidation_df = pd.read_csv(consolidation_path)

# Standardize column names
consolidation_df = consolidation_df.rename(columns={
    "SMILES": "smiles",
    "Solvent": "solvent"
})

fluodb_df = basic_clean(
    fluodb_df,
    smiles_col='smiles',
    target_col='abs'
)

fluodb_df = fluodb_df.rename(columns={'abs':'peakwavs_max'})


# -----------------------
# Clean Consolidation dataset
# -----------------------

consolidation_df = basic_clean(
    consolidation_df,
    smiles_col='smiles',
    target_col='Ex (nm)'
)

consolidation_df = consolidation_df.rename(columns={
    'Ex (nm)': 'peakwavs_max'
})


# Optional duplicate aggregation
fluodb_df = handle_duplicates(fluodb_df, target_col='peakwavs_max')
consolidation_df = handle_duplicates(consolidation_df, target_col='peakwavs_max')

for dataset_name, dataset_df in [
    ("fluodb", fluodb_df),
    ("consolidation", consolidation_df)
]:

    for split_type in ['random', 'group_by_smiles', 'scaffold']:

        output_dir = os.path.join(
            DATA_DIR,
            f"splits/{dataset_name}/{split_type}"
        )

        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

        _, _, _ = data_split_and_write(
            dataset_df,
            feature_names=None,
            target_names=['peakwavs_max'],
            solvation=False,
            split_type=split_type,
            scale_targets=False,
            write_files=True,
            random_seed=0
        )