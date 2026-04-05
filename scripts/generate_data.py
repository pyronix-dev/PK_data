"""
generate_data.py
Generates the complete PK Predictor Challenge dataset.

Usage:
    # Training data
    python generate_data.py --mode train --seed 42
    
    # Test data
    python generate_data.py --mode test --seed 123
    
    # Quick test (small dataset)
    python generate_data.py --mode train --seed 42 --quick
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from pbpk_simulator import (
    PhysiologyModel,
    QSPRPredictor,
    PBPKModel,
    PKParameterExtractor,
    add_measurement_noise,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

TIME_POINTS = [0, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48, 72]

DOSES_MG = [10, 25, 50, 100, 200, 500]
DOSE_PROBS = [0.1, 0.2, 0.25, 0.25, 0.15, 0.05]

ROUTE_PROBS = {'oral': 0.7, 'iv': 0.3}

# Drug class definitions with descriptor distributions
DRUG_CLASSES = {
    'benzodiazepine': {
        'mw': (280, 40),
        'logp': (2.5, 0.8),
        'tpsa': (35, 15),
        'hbd': (1, 0.5),
        'hba': (3, 1),
        'rotatable_bonds': (2, 1),
        'aromatic_rings': (2, 0.5),
        'pka_basic': (3.5, 0.5),
        'n_samples': 35,
    },
    'beta_blocker': {
        'mw': (260, 50),
        'logp': (2.0, 0.8),
        'tpsa': (50, 15),
        'hbd': (2, 0.5),
        'hba': (4, 1),
        'rotatable_bonds': (5, 1.5),
        'aromatic_rings': (1, 0.5),
        'pka_basic': (9.5, 0.5),
        'n_samples': 25,
    },
    'statin': {
        'mw': (400, 60),
        'logp': (3.5, 1.0),
        'tpsa': (100, 25),
        'hbd': (3, 1),
        'hba': (6, 1.5),
        'rotatable_bonds': (7, 2),
        'aromatic_rings': (2, 0.5),
        'pka_acidic': (4.5, 0.5),
        'n_samples': 25,
    },
    'fluoroquinolone': {
        'mw': (330, 40),
        'logp': (1.0, 0.5),
        'tpsa': (70, 15),
        'hbd': (2, 0.5),
        'hba': (5, 1),
        'rotatable_bonds': (2, 1),
        'aromatic_rings': (2, 0.5),
        'pka_basic': (8.5, 0.5),
        'pka_acidic': (6.0, 0.5),
        'n_samples': 25,
    },
    'nsaid': {
        'mw': (230, 40),
        'logp': (3.0, 0.8),
        'tpsa': (40, 15),
        'hbd': (1, 0.5),
        'hba': (2, 0.5),
        'rotatable_bonds': (2, 1),
        'aromatic_rings': (2, 0.5),
        'pka_acidic': (4.5, 0.5),
        'n_samples': 35,
    },
    'ace_inhibitor': {
        'mw': (380, 50),
        'logp': (1.5, 0.8),
        'tpsa': (80, 20),
        'hbd': (3, 1),
        'hba': (5, 1.5),
        'rotatable_bonds': (6, 2),
        'aromatic_rings': (1, 0.5),
        'pka_acidic': (3.5, 0.5),
        'n_samples': 25,
    },
    'ssri': {
        'mw': (310, 50),
        'logp': (3.5, 1.0),
        'tpsa': (25, 10),
        'hbd': (1, 0.5),
        'hba': (2, 0.5),
        'rotatable_bonds': (3, 1.5),
        'aromatic_rings': (2, 0.5),
        'pka_basic': (9.0, 0.5),
        'n_samples': 25,
    },
    'opioid': {
        'mw': (300, 40),
        'logp': (2.0, 0.8),
        'tpsa': (40, 15),
        'hbd': (2, 0.5),
        'hba': (3, 0.5),
        'rotatable_bonds': (2, 1),
        'aromatic_rings': (1, 0.5),
        'pka_basic': (8.5, 0.5),
        'n_samples': 25,
    },
}

NOVEL_SCAFFOLD_COUNT = 150

# Known drug SMILES for generating realistic molecules
KNOWN_DRUG_SMILES = {
    'benzodiazepine': [
        'CN1C(=O)CN=C(C2=C1C=CC=C2)C3=CC=CC=C3Cl',  # Diazepam
        'C1=CC=C(C=C1)C2=NC3=C(N2CCOCCO)C4=C(C=CC=C4)Cl',  # Clonazepam
        'CC1=NN=C2C(=O)N(C(=NC2=C1C3=CC=CC=C3)C)C',  # Triazolam
    ],
    'beta_blocker': [
        'CC(C)NCC(C1=CC2=CC=CC=C2O1)O',  # Propranolol
        'COC1=CC=C(C=C1)CNC(C)OC2=CC=CC=C2',  # Metoprolol
        'CC(C)NCC(C1=CC=C(C=C1)O)O',  # Atenolol-like
    ],
    'statin': [
        'CC(C)C1=C(C=C(C=C1)C(C)C)C2=CC(=C(N2)C3=CC=C(C=C3)F)C4=CC=C(C=C4)F',  # Atorvastatin-like
        'CC1=CC(=C2C(=C1)C=CC(=C2OC(=O)C[C@@H](O)C[C@@H](O)CC(=O)O)C3=CC=C(C=C3)F)C',  # Rosuvastatin-like
    ],
    'fluoroquinolone': [
        'CN1CCN(CC1)C2=C(C3=CC(=C(C=C3F)C2=O)C(=O)O)N4CCN(CC4)C',  # Ciprofloxacin-like
    ],
    'nsaid': [
        'CC(C)C1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
        'COC1=CC=C2C(=C1)C=C(C=C2)C(C)C(=O)O',  # Naproxen
        'OC(=O)CC1=CC=CC=C1NC2=C(Cl)C=CC=C2Cl',  # Diclofenac
    ],
    'ace_inhibitor': [
        'CC(C)(C)CC(C(=O)N1CCCC1C(=O)O)NC(C(=O)O)CC2=CC=CC=C2',  # Ramipril-like
    ],
    'ssri': [
        'CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F',  # Fluoxetine
        'CN[C@H]1CC[C@@H](C2=CC=CC=C2O1)C3=CC=C(C=C3)Cl',  # Sertraline-like
    ],
    'opioid': [
        'CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O',  # Morphine
        'COc1ccc2c3c1OC[C@@H]2[C@H]4Cc5c(OC)ccc(O)c5[C@]3(CCN4C)O',  # Codeine-like
    ],
}

OUTPUT_DIR = Path(__file__).parent / 'data'


# ============================================================
# MOLECULE GENERATION
# ============================================================

def generate_molecules(n_molecules: int, rng: np.random.Generator,
                       include_known: bool = True) -> pd.DataFrame:
    """
    Generate a diverse population of drug-like molecules.
    
    Args:
        n_molecules: Number of molecules to generate
        rng: Random number generator
        include_known: Whether to include perturbed known drugs
    
    Returns:
        DataFrame with molecule descriptors
    """
    records = []
    mol_idx = 0
    
    if include_known:
        # Generate perturbed known drugs
        for drug_class, params in DRUG_CLASSES.items():
            n_class = params['n_samples']
            smiles_list = KNOWN_DRUG_SMILES.get(drug_class, ['CCO'])
            
            for _ in range(n_class):
                # Perturb descriptors around class means
                record = {
                    'molecule_id': f'mol_{mol_idx:04d}',
                    'drug_class': drug_class,
                    'is_known_drug': 1,
                }
                
                for key in ['mw', 'logp', 'tpsa', 'hbd', 'hba', 
                            'rotatable_bonds', 'aromatic_rings']:
                    if key in params:
                        mu, sigma = params[key]
                        val = rng.normal(mu, sigma)
                        if key in ['hbd', 'hba', 'rotatable_bonds', 'aromatic_rings']:
                            val = max(0, round(val))
                        elif key == 'mw':
                            val = max(50, val)
                        record[key] = val
                
                # pKa values
                if 'pka_basic' in params:
                    record['pka_basic'] = max(1.0, rng.normal(*params['pka_basic']))
                else:
                    record['pka_basic'] = np.nan
                
                if 'pka_acidic' in params:
                    record['pka_acidic'] = max(1.0, rng.normal(*params['pka_acidic']))
                else:
                    record['pka_acidic'] = np.nan
                
                # Derived properties
                record['fraction_ionized_ph74'] = _compute_ionization(
                    record.get('pka_basic', np.nan),
                    record.get('pka_acidic', np.nan)
                )
                record['fraction_unbound'] = np.clip(
                    rng.beta(2, 5) + 0.1 * (5 - record.get('logp', 2)),
                    0.01, 1.0
                )
                
                # Placeholder SMILES (in real generation, would use actual structure)
                base_smiles = rng.choice(smiles_list)
                record['smiles'] = base_smiles  # Would be perturbed in full version
                
                # Morgan fingerprint (placeholder - would be computed from actual SMILES)
                record['morgan_fp_256'] = _generate_random_fp(256, rng)
                
                # Similarity to known drugs
                record['similarity_to_known'] = rng.uniform(0.5, 0.95)
                
                records.append(record)
                mol_idx += 1
    
    # Generate novel scaffolds
    n_novel = n_molecules - mol_idx
    for _ in range(n_novel):
        record = {
            'molecule_id': f'mol_{mol_idx:04d}',
            'drug_class': 'novel_scaffold',
            'is_known_drug': 0,
        }
        
        # Sample from broad drug-like distribution
        record['mw'] = max(50, rng.lognormal(5.5, 0.5))
        record['logp'] = rng.normal(2.5, 1.5)
        record['tpsa'] = max(0, rng.exponential(70))
        record['hbd'] = max(0, rng.poisson(2))
        record['hba'] = max(0, rng.poisson(5))
        record['rotatable_bonds'] = max(0, rng.poisson(4))
        record['aromatic_rings'] = max(0, min(8, rng.poisson(2)))
        
        # pKa
        if rng.random() > 0.3:
            record['pka_basic'] = max(1.0, rng.normal(8.0, 2.5))
        else:
            record['pka_basic'] = np.nan
        
        if rng.random() > 0.5:
            record['pka_acidic'] = max(1.0, rng.normal(4.5, 1.5))
        else:
            record['pka_acidic'] = np.nan
        
        record['fraction_ionized_ph74'] = _compute_ionization(
            record.get('pka_basic', np.nan),
            record.get('pka_acidic', np.nan)
        )
        record['fraction_unbound'] = np.clip(
            rng.beta(2, 5),
            0.01, 1.0
        )
        
        record['smiles'] = f'NOVEL_{mol_idx}'
        record['morgan_fp_256'] = _generate_random_fp(256, rng)
        
        # Low similarity to known drugs (by definition)
        record['similarity_to_known'] = rng.uniform(0.05, 0.35)
        
        records.append(record)
        mol_idx += 1
    
    df = pd.DataFrame(records)
    return df


def _compute_ionization(pka_basic: float, pka_acidic: float) -> float:
    """Compute fraction ionized at blood pH 7.4."""
    ph = 7.4
    
    if not np.isnan(pka_basic) and not np.isnan(pka_acidic):
        # Both basic and acidic groups
        basic_ionized = 1 / (1 + 10 ** (ph - pka_basic))
        acidic_ionized = 1 / (1 + 10 ** (pka_acidic - ph))
        return (basic_ionized + acidic_ionized) / 2
    elif not np.isnan(pka_basic):
        return 1 / (1 + 10 ** (ph - pka_basic))
    elif not np.isnan(pka_acidic):
        return 1 / (1 + 10 ** (pka_acidic - ph))
    else:
        return 0.0  # Neutral molecule


def _generate_random_fp(n_bits: int, rng: np.random.Generator) -> str:
    """Generate a random Morgan fingerprint as hex string."""
    # In real implementation, this would be computed from actual SMILES
    # For synthetic data, generate a random bit pattern
    bits = rng.integers(0, 2, size=n_bits)
    # Convert to hex
    int_val = int(''.join(str(b) for b in bits), 2)
    return f'{int_val:0{n_bits // 4}x}'


# ============================================================
# PATIENT GENERATION
# ============================================================

def generate_patients(n_patients: int, rng: np.random.Generator,
                      extreme_fraction: float = 0.15) -> pd.DataFrame:
    """
    Generate a diverse patient population.
    
    Args:
        n_patients: Number of patients
        rng: Random number generator
        extreme_fraction: Fraction with extreme physiology
    
    Returns:
        DataFrame with patient parameters
    """
    records = []
    n_extreme = int(n_patients * extreme_fraction)
    n_normal = n_patients - n_extreme
    
    # Normal patients
    for _ in range(n_normal):
        record = _generate_normal_patient(rng, is_extreme=False)
        records.append(record)
    
    # Extreme patients
    for _ in range(n_extreme):
        record = _generate_normal_patient(rng, is_extreme=True)
        records.append(record)
    
    df = pd.DataFrame(records)
    df['patient_id'] = [f'pat_{i:04d}' for i in range(len(df))]
    
    return df


def _generate_normal_patient(rng: np.random.Generator, 
                              is_extreme: bool = False) -> Dict:
    """Generate a single patient profile."""
    
    if is_extreme:
        # Sample from distribution tails
        age = int(rng.choice([18, 19, 20, 80, 81, 82, 83, 84, 85]))
        weight = rng.choice(
            [rng.lognormal(3.8, 0.1),  # Very light
             rng.lognormal(5.2, 0.2)]   # Very heavy
        )
        height = rng.choice(
            [rng.normal(148, 3),  # Very short
             rng.normal(200, 5)]   # Very tall
        )
        hepatic_impairment = rng.choice([0.75, 1.0], p=[0.5, 0.5])
        renal_impairment = rng.choice([0.3, 0.5], p=[0.3, 0.7])
    else:
        # Normal distribution
        # Age: weighted toward 40-70
        age = int(rng.choice(
            np.arange(18, 86),
            p=_age_distribution_weights()
        ))
        weight = rng.lognormal(4.3, 0.15)  # ~73 kg median
        height = rng.normal(170, 8)
        hepatic_impairment = rng.choice(
            [0, 0, 0, 0, 0.5],
            p=[0.2, 0.2, 0.2, 0.2, 0.2]
        )
        renal_impairment = rng.choice(
            [0.7, 0.85, 1.0],
            p=[0.1, 0.2, 0.7]
        )
    
    sex = rng.choice([0, 1])
    
    # Correlated lab values
    albumin = rng.normal(4.0, 0.4)
    if is_extreme and rng.random() > 0.5:
        albumin = rng.normal(2.5, 0.3)  # Hypoalbuminemia
    
    aag = rng.normal(1.0, 0.2)
    gastric_ph = rng.normal(1.5, 0.4)
    transit_time = rng.normal(3.0, 0.8)
    hematocrit = rng.normal(0.42, 0.04)
    cardiac_output = rng.normal(5.5, 0.8)  # L/min
    
    # CYP genotypes
    cyp3a4 = rng.choice(['PM', 'IM', 'NM', 'UM'], p=[0.02, 0.1, 0.83, 0.05])
    cyp2d6 = rng.choice(['PM', 'IM', 'NM', 'UM'], p=[0.07, 0.1, 0.73, 0.1])
    cyp2c9 = rng.choice(['PM', 'IM', 'NM', 'UM'], p=[0.05, 0.15, 0.70, 0.1])
    
    bmi = weight / (height / 100) ** 2
    bsa = 0.007184 * (weight ** 0.425) * (height ** 0.725)
    
    return {
        'age': age,
        'sex': sex,
        'weight_kg': round(weight, 1),
        'height_cm': round(height, 1),
        'bmi': round(bmi, 1),
        'bsa_m2': round(bsa, 2),
        'hepatic_impairment': hepatic_impairment,
        'renal_impairment': renal_impairment,
        'cyp2d6_status': cyp2d6,
        'cyp3a4_status': cyp3a4,
        'cyp2c9_status': cyp2c9,
        'albumin_gdl': round(albumin, 1),
        'alpha1_acid_glycoprotein': round(aag, 2),
        'gastric_ph': round(gastric_ph, 2),
        'intestinal_transit_time_h': round(max(1.0, transit_time), 1),
        'hematocrit': round(hematocrit, 2),
        'cardiac_output_l_min': round(max(3.0, cardiac_output), 1),
        'is_extreme_physiology': is_extreme,
    }


def _age_distribution_weights() -> np.ndarray:
    """Weights for age sampling (clinical trial demographics)."""
    weights = np.zeros(68)  # ages 18-85
    weights[0:10] = 0.02    # 18-27
    weights[10:25] = 0.04   # 28-42
    weights[25:40] = 0.05   # 43-57
    weights[40:55] = 0.03   # 58-72
    weights[55:] = 0.01     # 73-85
    return weights / weights.sum()


# ============================================================
# SAMPLE GENERATION
# ============================================================

def generate_samples(molecules_df: pd.DataFrame,
                     patients_df: pd.DataFrame,
                     n_samples: int,
                     rng: np.random.Generator,
                     is_test: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate molecule-patient-dose samples with concentration-time curves.
    
    Args:
        molecules_df: Molecule descriptors
        patients_df: Patient parameters
        n_samples: Number of samples
        rng: Random number generator
        is_test: Whether generating test set (includes more extrapolation)
    
    Returns:
        curves_df, pk_params_df, sample_map_df
    """
    qsp = QSPRPredictor(rng)
    extractor = PKParameterExtractor()
    
    all_curves = []
    all_pk_params = []
    all_sample_map = []
    
    # Pre-compute extrapolation molecule indices
    if is_test:
        novel_mask = molecules_df['drug_class'] == 'novel_scaffold'
        low_sim = molecules_df['similarity_to_known'] < 0.35
        extrapolation_mol_idx = set(
            molecules_df[novel_mask | low_sim].index
        )
    else:
        extrapolation_mol_idx = set()
    
    for i in tqdm(range(n_samples), desc='Generating samples'):
        # Sample molecule
        if is_test and rng.random() < 0.25:
            # Force extrapolation molecules in test set
            mol_idx = rng.choice(list(extrapolation_mol_idx))
        else:
            mol_idx = rng.integers(0, len(molecules_df))
        
        mol = molecules_df.iloc[mol_idx]
        
        # Sample patient
        if is_test and rng.random() < 0.12:
            # Force extreme patients in test set
            extreme_mask = patients_df['is_extreme_physiology'] == 1
            pat_idx = rng.choice(patients_df[extreme_mask].index)
        else:
            pat_idx = rng.integers(0, len(patients_df))
        
        patient = patients_df.iloc[pat_idx]
        
        # Sample dose and route
        dose_mg = rng.choice(DOSES_MG, p=DOSE_PROBS)
        route = rng.choice(list(ROUTE_PROBS.keys()), p=list(ROUTE_PROBS.values()))
        
        # Generate sample ID
        sample_id = f'sample_{i:06d}'
        
        # Build physiology model
        physiology = PhysiologyModel(patient.to_dict())
        
        # Predict PK parameters from structure
        ka = qsp.predict_absorption_rate(mol.to_dict(), physiology)
        fu = qsp.predict_fraction_unbound(mol.to_dict(), physiology)
        cl_int = qsp.predict_intrinsic_clearance(mol.to_dict(), patient.to_dict())
        kp = qsp.predict_tissue_partitioning(mol.to_dict())
        
        # Compute hepatic clearance (well-stirred model)
        q_h = physiology.blood_flows.get('liver', 90.0)
        cl_hepatic = q_h * (fu * cl_int) / (q_h + fu * cl_int)
        
        # Renal clearance
        gfr = physiology.get_gfr()
        cl_renal = gfr * fu
        
        # Bioavailability (oral)
        f_oral = qsp.predict_bioavailability(mol.to_dict(), ka, cl_hepatic, physiology)
        
        pk_params = {
            'ka': ka,
            'cl_hepatic': cl_hepatic,
            'cl_renal': cl_renal,
            'fu': fu,
            'kp': kp,
            'f_oral': f_oral,
        }
        
        # Run simulation
        pbpk = PBPKModel(physiology)
        sol, _ = pbpk.simulate(pk_params, dose_mg, route, TIME_POINTS)
        
        # Extract concentrations
        V = physiology.organ_volumes
        measured_compartments = ['blood', 'liver', 'kidney', 'brain', 'fat']
        
        for comp in measured_compartments:
            comp_idx = PBPKModel.COMPARMENT_IDX[comp]
            conc_true = sol.y[comp_idx] / V[comp] * 1e6  # mg/L → ng/mL
            
            # Add measurement noise
            conc_noisy = add_measurement_noise(conc_true, rng=rng)
            
            for t, c in zip(TIME_POINTS, conc_noisy):
                all_curves.append({
                    'sample_id': sample_id,
                    'molecule_id': mol['molecule_id'],
                    'patient_id': patient['patient_id'],
                    'compartment': comp,
                    'time_h': t,
                    'concentration_ng_ml': round(c, 4),
                    'dose_mg': dose_mg,
                    'route': route,
                })
        
        # Compute PK parameters from true (non-noisy) curves
        true_curves_data = []
        for comp in measured_compartments:
            comp_idx = PBPKModel.COMPARMENT_IDX[comp]
            conc_true = sol.y[comp_idx] / V[comp] * 1e6
            for t, c in zip(TIME_POINTS, conc_true):
                true_curves_data.append({
                    'compartment': comp,
                    'time_h': t,
                    'concentration_ng_ml': c,
                })
        
        true_curves_df = pd.DataFrame(true_curves_data)
        pk_summary = extractor.extract_parameters(
            true_curves_df, pk_params, dose_mg, route
        )
        pk_summary['sample_id'] = sample_id
        all_pk_params.append(pk_summary)
        
        # Sample map
        is_extrap_mol = mol_idx in extrapolation_mol_idx or mol['similarity_to_known'] < 0.4
        is_extrap_pat = patient['is_extreme_physiology'] == 1
        
        all_sample_map.append({
            'sample_id': sample_id,
            'molecule_id': mol['molecule_id'],
            'patient_id': patient['patient_id'],
            'dose_mg': dose_mg,
            'route': route,
            'is_extrapolation_molecule': is_extrap_mol,
            'is_extrapolation_patient': is_extrap_pat,
            'is_drug_interaction': False,  # Would be True for CYP inhibitor scenarios
        })
    
    curves_df = pd.DataFrame(all_curves)
    pk_params_df = pd.DataFrame(all_pk_params)
    sample_map_df = pd.DataFrame(all_sample_map)
    
    return curves_df, pk_params_df, sample_map_df


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate PK Challenge dataset')
    parser.add_argument('--mode', choices=['train', 'test', 'all'], default='all',
                        help='Generate training data, test data, or everything at once')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n-molecules', type=int, default=500,
                        help='Number of molecules')
    parser.add_argument('--n-patients', type=int, default=200,
                        help='Number of patients')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Number of samples (default: 15000 train, 3000 test, 18000 all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 100 molecules, 50 patients, 500 samples')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    # Setup
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        n_molecules = 100
        n_patients = 50
        n_samples = 500
        prefix = 'quick'
    else:
        n_molecules = args.n_molecules
        n_patients = args.n_patients
        if args.mode == 'all':
            n_samples = args.n_samples or 18000
            prefix = 'all'
        elif args.mode == 'train':
            n_samples = args.n_samples or 15000
            prefix = 'train'
        else:
            n_samples = args.n_samples or 3000
            prefix = 'test'

    logger.info(f"Generating {args.mode} dataset:")
    logger.info(f"  Molecules: {n_molecules}")
    logger.info(f"  Patients: {n_patients}")
    logger.info(f"  Samples: {n_samples}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Output: {output_dir}")

    # Generate molecules
    logger.info("Generating molecules...")
    molecules_df = generate_molecules(n_molecules, rng, include_known=True)
    molecules_df.to_csv(output_dir / 'molecules.csv', index=False)
    logger.info(f"  Saved molecules.csv ({len(molecules_df)} rows)")

    # Generate patients
    logger.info("Generating patients...")
    patients_df = generate_patients(n_patients, rng)
    patients_df.to_csv(output_dir / 'patients.csv', index=False)
    logger.info(f"  Saved patients.csv ({len(patients_df)} rows)")

    # Generate samples
    logger.info("Generating PK samples...")
    is_test = (args.mode == 'test')
    curves_df, pk_params_df, sample_map_df = generate_samples(
        molecules_df, patients_df, n_samples, rng,
        is_test=is_test
    )

    # Save individual files
    curves_df.to_parquet(output_dir / f'{prefix}_curves.parquet')
    pk_params_df.to_csv(output_dir / f'{prefix}_pk_params.csv', index=False)
    sample_map_df.to_csv(output_dir / f'{prefix}.csv', index=False)

    # Save combined data.csv (curves + PK params merged) — used by prepare.py
    combined = curves_df.merge(pk_params_df, on='sample_id', how='left')
    combined.to_csv(output_dir / 'data.csv', index=False)

    logger.info(f"  Saved {prefix}_curves.parquet ({len(curves_df)} rows)")
    logger.info(f"  Saved {prefix}_pk_params.csv ({len(pk_params_df)} rows)")
    logger.info(f"  Saved {prefix}.csv ({len(sample_map_df)} rows)")
    logger.info(f"  Saved data.csv ({len(combined)} rows)")
    
    # Summary statistics
    logger.info("\n=== Dataset Summary ===")
    logger.info(f"Molecules: {len(molecules_df)}")
    logger.info(f"  Known drugs: {(molecules_df['is_known_drug'] == 1).sum()}")
    logger.info(f"  Novel scaffolds: {(molecules_df['drug_class'] == 'novel_scaffold').sum()}")
    logger.info(f"Patients: {len(patients_df)}")
    logger.info(f"  Extreme physiology: {patients_df['is_extreme_physiology'].sum()}")
    logger.info(f"Samples: {n_samples}")
    logger.info(f"  Oral: {(sample_map_df['route'] == 'oral').sum()}")
    logger.info(f"  IV: {(sample_map_df['route'] == 'iv').sum()}")
    logger.info(f"  Extrapolation molecules: {sample_map_df['is_extrapolation_molecule'].sum()}")
    logger.info(f"  Extrapolation patients: {sample_map_df['is_extrapolation_patient'].sum()}")
    logger.info(f"Concentration measurements: {len(curves_df)}")
    
    if args.mode == 'test':
        # Generate sample submission template
        _generate_sample_submission(sample_map_df, output_dir)
    
    logger.info("\nDataset generation complete!")


def _generate_sample_submission(sample_map_df: pd.DataFrame, output_dir: Path):
    """Generate sample submission template."""
    records = []
    
    for _, row in sample_map_df.iterrows():
        sample_id = row['sample_id']
        dose = row['dose_mg']
        
        for comp in ['blood', 'liver', 'kidney', 'brain', 'fat']:
            for t in TIME_POINTS:
                # Placeholder: simple exponential decay
                if row['route'] == 'oral':
                    c = dose * 0.5 * np.exp(-0.1 * t) * (1 - np.exp(-2 * t))
                else:
                    c = dose * np.exp(-0.1 * t)
                c = max(c, 0.01)
                
                records.append({
                    'sample_id': sample_id,
                    'compartment': comp,
                    'time_h': t,
                    'concentration_ng_ml': round(c, 4),
                    'cmax_blood': dose * 0.5,
                    'tmax_blood': 2.0,
                    'auc_0_72_blood': dose * 5.0,
                    'half_life': 6.9,
                    'clearance_l_h': 15.0,
                })
    
    submission_df = pd.DataFrame(records)
    submission_df.to_csv(output_dir / 'sample_submission.csv', index=False)
    logger.info(f"  Saved sample_submission.csv ({len(submission_df)} rows)")


if __name__ == '__main__':
    main()
