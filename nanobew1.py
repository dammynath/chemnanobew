"""
nanobew_app.py – Synthesis Optimization Suite (RDKit‑Mode)
Updated with CIS-Te/ZnS experimental data (Dr. Adimula, 2026)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import qmc
import plotly.io as pio
import base64
import subprocess
import os
import tempfile
import toml
import io
from pathlib import Path
import sys
import time
from datetime import datetime
from itertools import combinations, product
import warnings
import pickle
import threading
import requests
from functools import wraps
import traceback
warnings.filterwarnings('ignore')

# Optional imports
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# RDKit imports - comprehensive
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, AllChem, rdMolDescriptors
    from rdkit.Chem.Draw import MolsToGridImage
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.Fingerprints import FingerprintMols
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem import rdMolTransforms
    from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
    from rdkit.Chem import rdDepictor
    from rdkit.Chem import rdDistGeom
    from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors, NumRotatableBonds
    from rdkit.Chem.QED import qed
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import Crippen
    from rdkit.Chem import GraphDescriptors
    from rdkit.Chem import Descriptors3D
    from rdkit.Chem import rdMolAlign
    from rdkit.DataStructs import TanimotoSimilarity
    from rdkit.Chem import rdChemReactions
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem import SaltRemover
    from rdkit.Chem import rdRGroupDecomposition
    from rdkit.Chem import rdMMPA
    
    RDKIT_AVAILABLE = True
    st.sidebar.success("✅ RDKit fully loaded with all modules")
except ImportError as e:
    RDKIT_AVAILABLE = False
    st.error(f"❌ RDKit import error: {str(e)}")
    st.stop()

# Optional OpenAI import
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# Bayesian Optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# Deep Learning (optional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================================
# EXPERIMENTAL DATA FROM DR. ADIMULA (2026) - CIS-Te/ZnS Quantum Dots
# ============================================================================
# Optimized conditions achieving:
# - Emission: 815 nm with intensity 12,828.54
# - Quantum Yield: 5.13%
# - Lifetime: 37.54 ns
# - Excitation: 550 nm

EXPERIMENTAL_OPTIMUM = {
    'cucl2_mg': 11.0,
    'incl3_mg': 55.0,
    'trisodium_citrate_mg': 294.0,
    'tga_ul': 70.0,
    'na2s_mg': 97.5,
    'nabh4_mg': 26.0,
    'te_salt_mg': 1.7,
    'zn_precursor_mg': 30.0,
    'thiourea_mg': 15.0,
    'core_temp_c': 95.0,
    'core_time_min': 30.0,
    'te_incorp_time_min': 10.0,
    'shell_time_min': 20.0,
    'pH': 3.0,
    'excitation_nm': 550.0,
    'emission_nm': 815.0,
    'pl_intensity': 12828.54,
    'lifetime_ns': 37.54,
    'quantum_yield_percent': 5.13
}

EXPERIMENTAL_PARAM_RANGES = {
    'cucl2_mg': (8.0, 14.0),
    'incl3_mg': (40.0, 70.0),
    'te_salt_mg': (1.0, 2.5),
    'zn_precursor_mg': (20.0, 40.0),
    'core_temp_c': (85.0, 105.0),
    'core_time_min': (20, 40),
    'pH': (2.5, 4.0),
    'shell_time_min': (10, 30),
    'te_incorp_time_min': (5, 20),
    'thiourea_mg': (10.0, 20.0),
    'na2s_mg': (80.0, 120.0),
    'tga_ul': (50.0, 90.0)
}

# ============================================================================
# LOGO HANDLING
# ============================================================================

def get_logo_base64():
    """Convert logo image to base64 for embedding"""
    try:
        logo_path = "images/chemnanobew_icon.png"
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        pass
    return None

def render_logo():
    """Render the CHEM-NANO-BEW logo"""
    logo_base64 = get_logo_base64()
    
    if logo_base64:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem;'>
            <img src='data:image/png;base64,{logo_base64}' style='max-width: 200px;'>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 1rem;'>
            <h1 style='color: white; font-size: 2rem; margin: 0;'>CHEM-NANO-BEW</h1>
            <p style='color: white; font-size: 1rem; margin: 0;'>LABORATORY</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# Utility Functions
# ============================================================================
def normalize_series(series):
    """Normalize a pandas series to [0,1] range"""
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

def calculate_composite_score(df, weights):
    """Calculate composite score from multiple objectives"""
    score = 0
    for prop, weight in weights.items():
        if prop in df.columns:
            score += weight * normalize_series(df[prop])
    return score

def save_uploaded_image(uploaded_file):
    """Save uploaded image to images folder"""
    if uploaded_file is not None:
        os.makedirs("images", exist_ok=True)
        path = os.path.join("images", uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return path
    return None

# ============================================================================
# Page config (must be first)
# ============================================================================
st.set_page_config(
    page_title="CHEM-NANO-BEW Laboratory",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS
# ============================================================================
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #2c3e50; text-align: center; margin-bottom: 0.5rem; padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .sub-header { font-size: 1.5rem; font-weight: 600; color: #34495e; margin-top: 1rem; margin-bottom: 1rem;
        padding-bottom: 0.5rem; border-bottom: 2px solid #3498db; }
    .info-box { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3498db; margin-bottom: 1rem; }
    .sidebar-logo { text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 0.5rem; }
    .sidebar-logo-text { color: white; font-size: 1.2rem; font-weight: 600; }
    .lab-subtitle { text-align: center; color: #7f8c8d; font-size: 1rem; margin-bottom: 2rem; }
    .footer { text-align: center; padding: 2rem; color: #7f8c8d; border-top: 1px solid #ecf0f1; margin-top: 3rem; }
    .success-box { background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #28a745; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DataManager Class
# ============================================================================
class DataManager:
    @staticmethod
    def load_data(uploaded_file):
        if uploaded_file is not None:
            try:
                return pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return None
        return None

    @staticmethod
    def create_sample_qd_data(n=50):
        np.random.seed(42)
        return pd.DataFrame({
            'precursor_ratio': np.random.uniform(0.5, 2.0, n),
            'temperature': np.random.uniform(150, 250, n),
            'reaction_time': np.random.uniform(30, 180, n),
            'zn_precursor': np.random.uniform(0.1, 1.0, n),
            'pH': np.random.uniform(4, 10, n),
            'surfactant': np.random.choice(['oleic_acid', 'oleylamine', 'dodecanethiol'], n),
            'solvent': np.random.choice(['octadecene', 'toluene', 'chloroform'], n),
            'absorption_nm': np.random.normal(700, 100, n),
            'plqy_percent': np.random.normal(50, 15, n),
            'pce_percent': np.random.normal(45, 12, n),
            'soq_au': np.random.normal(0.5, 0.15, n)
        })

    @staticmethod
    def create_sample_porphyrin_data(n=50):
        np.random.seed(42)
        return pd.DataFrame({
            'aldehyde_conc': np.random.uniform(0.01, 0.1, n),
            'pyrrole_conc': np.random.uniform(0.01, 0.1, n),
            'temperature': np.random.uniform(20, 150, n),
            'reaction_time': np.random.uniform(30, 1440, n),
            'catalyst_conc': np.random.uniform(0.001, 0.05, n),
            'catalyst_type': np.random.choice(['BF3', 'TFA', 'DDQ', 'p-chloranil'], n),
            'solvent': np.random.choice(['DCM', 'CHCl3', 'toluene', 'DMF'], n),
            'yield_percent': np.random.normal(45, 15, n),
            'purity_percent': np.random.normal(85, 8, n),
            'singlet_oxygen_au': np.random.normal(0.5, 0.15, n),
            'fluorescence_qy': np.random.normal(0.12, 0.05, n)
        })

# ============================================================================
# MOLECULAR UTILITIES
# ============================================================================

class MolecularUtils:
    """Molecular utilities with RDKit support"""
    
    def __init__(self):
        self.rdkit = RDKIT_AVAILABLE
        self.base_abs = 410
        self.base_fluor = 630
        self.base_qy = 0.12
        
        self.known_porphyrins = [
            ("C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=N5)C=C2", "Unsubstituted", 0, 0, 0.00),
            ("C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(Br)=N5)C=C2", "Bromo", 15, 10, -0.02),
            ("C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(Cl)=N5)C=C2", "Chloro", 8, 5, -0.01),
            ("C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(I)=N5)C=C2", "Iodo", 25, 20, -0.05),
            ("C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(OC)=N5)C=C2", "Methoxy", 20, 15, 0.03),
            ("C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(C)=N5)C=C2", "Methyl", 5, 3, 0.01),
            ("C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(N)=N5)C=C2", "Amino", 30, 25, 0.08),
        ]
    
    def generate_porphyrin_variants(self, n=10, target_abs=None, target_fluor=None, target_qy=None):
        scored = []
        for smi, name, delta_abs, delta_fluor, delta_qy in self.known_porphyrins:
            abs_wl = self.base_abs + delta_abs + np.random.normal(0, 5)
            fluor_wl = self.base_fluor + delta_fluor + np.random.normal(0, 8)
            qy = self.base_qy + delta_qy + np.random.normal(0, 0.02)
            qy = max(0, min(1, qy))
            
            score = 0
            if target_abs:
                score -= abs(abs_wl - target_abs) / 50.0
            if target_fluor:
                score -= abs(fluor_wl - target_fluor) / 50.0
            if target_qy:
                score -= abs(qy - target_qy) * 10.0
            
            scored.append((smi, score, abs_wl, fluor_wl, qy))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [smi for smi, _, _, _, _ in scored[:n]]
    
    def validate_smiles(self, smiles):
        if not smiles:
            return False
        if self.rdkit and Chem:
            try:
                mol = Chem.MolFromSmiles(smiles)
                return mol is not None
            except:
                return False
        return 'c' in smiles.lower() and 'n' in smiles.lower()
    
    def estimate_properties(self, smiles):
        if self.rdkit and Chem:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    return {
                        'molecular_weight': round(Descriptors.MolWt(mol), 2),
                        'logP': round(Descriptors.MolLogP(mol), 3),
                        'hba': Descriptors.NumHAcceptors(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                        'tpsa': round(Descriptors.TPSA(mol), 2),
                        'qed': round(Descriptors.qed(mol), 3),
                        'heavy_atoms': mol.GetNumHeavyAtoms(),
                        'rings': rdMolDescriptors.CalcNumRings(mol) if rdMolDescriptors else 0
                    }
            except Exception as e:
                pass
        
        c_count = smiles.lower().count('c')
        n_count = smiles.lower().count('n')
        o_count = smiles.lower().count('o')
        
        return {
            'molecular_weight': round(c_count*12 + n_count*14 + o_count*16 + 100, 2),
            'logP': round(-2 + c_count*0.3 - n_count*0.2, 3),
            'hba': n_count + o_count,
            'hbd': smiles.lower().count('oh'),
            'rotatable_bonds': smiles.count('=') // 2,
            'tpsa': round((n_count + o_count) * 12, 2),
            'qed': round(max(0, min(1, 0.3 + c_count*0.02 - n_count*0.01)), 3),
            'heavy_atoms': c_count + n_count + o_count,
            'rings': smiles.count('1')
        }


# ============================================================================
# QUANTUM DOT DATA MANAGER (UPDATED WITH CIS-Te/ZnS DATA)
# ============================================================================

class QDDataManager:
    """Manages data for different types of quantum dots and nanoparticles"""
    
    def __init__(self):
        self.qd_types = {
            'CIS/ZnS': {
                'description': 'Copper Indium Sulfide / Zinc Sulfide core/shell QDs',
                'optimal_absorption': '650-850 nm',
                'optimal_plqy': '50-80%',
                'key_params': ['cu_in_ratio', 'temperature', 'time', 'zn_precursor', 'pH']
            },
            'CIS-Te/ZnS': {
                'description': 'Copper Indium Sulfide-Tellurium / Zinc Sulfide core/shell QDs',
                'optimal_absorption': '800-900 nm (NIR)',
                'optimal_plqy': '5-8% (improving with optimization)',
                'optimal_emission': '815 nm maximum achieved',
                'optimal_intensity': '12,829 a.u.',
                'optimal_lifetime': '37.54 ns',
                'key_params': ['cucl2_mg', 'incl3_mg', 'te_salt_mg', 'zn_precursor_mg', 
                              'core_temp_c', 'core_time_min', 'pH', 'shell_time_min']
            },
            'AIS/ZnS': {
                'description': 'Silver Indium Sulfide / Zinc Sulfide core/shell QDs',
                'optimal_absorption': '550-750 nm',
                'optimal_plqy': '40-70%',
                'key_params': ['ag_in_ratio', 'temperature', 'time', 'zn_precursor', 'pH']
            },
            'CdSe/CdS': {
                'description': 'Cadmium Selenide / Cadmium Sulfide core/shell QDs',
                'optimal_absorption': '500-650 nm',
                'optimal_plqy': '60-90%',
                'key_params': ['cd_se_ratio', 'temperature', 'time', 'shell_thickness']
            },
            'Carbon Dots': {
                'description': 'Carbon-based fluorescent nanoparticles',
                'optimal_absorption': '350-500 nm',
                'optimal_plqy': '20-60%',
                'key_params': ['precursor_ratio', 'temperature', 'time', 'pH', 'microwave_power']
            },
            'Metal Nanoparticles': {
                'description': 'Au, Ag, Cu nanoparticles',
                'optimal_absorption': '400-600 nm (SPR)',
                'optimal_plqy': '1-10%',
                'key_params': ['metal_conc', 'reducing_agent', 'temperature', 'time', 'stabilizer']
            }
        }

# ============================================================================
# REINVENT4Wrapper Class
# ============================================================================

class REINVENT4Wrapper:
    """Wrapper class for REINVENT4 - handles RDKit fallback automatically"""
    
    def __init__(self):
        self.available = False
        self.rdkit_available = RDKIT_AVAILABLE
    
    def get_default_porphyrin_scaffold(self):
        return "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=N5)C=C2"
    
    def get_default_r_groups(self):
        return ["C", "CC", "CO", "CCO", "c1ccccc1", "Br", "Cl", "N", "C#N", "C(=O)O", "C(F)(F)F"]
    
    def get_default_linkers(self):
        return ["C", "CC", "C=CC", "C#C", "c1ccc2ccccc2c1"]


# ============================================================================
# CIS-Te/ZnS OPTIMIZER FUNCTIONS (UPDATED)
# ============================================================================

def generate_cis_te_data(n_samples=100):
    """Generate sample data for CIS-Te/ZnS quantum dots based on actual experimental results"""
    np.random.seed(42)
    
    data = {}
    
    # Generate parameters within experimental ranges
    for param, (low, high) in EXPERIMENTAL_PARAM_RANGES.items():
        data[param] = np.random.uniform(low, high, n_samples)
    
    # Ensure specific values for optimized sample
    opt_idx = 0
    for param, value in EXPERIMENTAL_OPTIMUM.items():
        if param in data:
            data[param][opt_idx] = value
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add excitation and absorption
    df['excitation_nm'] = 550 + np.random.normal(0, 15, n_samples)
    df['excitation_nm'] = np.clip(df['excitation_nm'], 500, 600)
    df['absorption_nm'] = df['excitation_nm'] - 30 + np.random.normal(0, 10, n_samples)
    df['absorption_nm'] = np.clip(df['absorption_nm'], 470, 570)
    
    # Calculate emission based on parameters
    # Te effect: red shift with more Te
    te_effect = (df['te_salt_mg'] - 1.7) / 1.7 * 15
    
    # Cu:In ratio effect
    cu_in_ratio = df['cucl2_mg'] / df['incl3_mg']
    ratio_effect = (cu_in_ratio - 0.2) / 0.2 * 10
    
    # pH effect (optimal at 3.0)
    ph_effect = -(df['pH'] - 3.0) ** 2 * 5
    
    # Temperature effect
    temp_effect = -(df['core_temp_c'] - 95) ** 2 / 100
    
    # Shell time effect
    shell_effect = (df['shell_time_min'] - 20) / 20 * 8
    
    df['emission_nm'] = 815 + te_effect + ratio_effect + ph_effect + temp_effect + shell_effect + np.random.normal(0, 5, n_samples)
    df['emission_nm'] = np.clip(df['emission_nm'], 750, 900)
    
    # Calculate PL intensity
    df['pl_intensity'] = 12828.54 + (df['zn_precursor_mg'] - 30) * 100 - (df['pH'] - 3.0)**2 * 500 + np.random.normal(0, 800, n_samples)
    df['pl_intensity'] = np.clip(df['pl_intensity'], 5000, 18000)
    
    # Calculate quantum yield
    df['quantum_yield_percent'] = 5.13 + (df['zn_precursor_mg'] - 30) * 0.08 - (df['te_salt_mg'] - 1.7) * 0.5 + np.random.normal(0, 0.5, n_samples)
    df['quantum_yield_percent'] = np.clip(df['quantum_yield_percent'], 2, 10)
    
    # Calculate lifetime
    df['lifetime_ns'] = 37.54 + (df['zn_precursor_mg'] - 30) * 0.15 + np.random.normal(0, 1.5, n_samples)
    df['lifetime_ns'] = np.clip(df['lifetime_ns'], 25, 50)
    
    # Calculate composite performance score
    df['composite_score'] = (
        normalize_series(df['emission_nm']) * 0.25 +
        normalize_series(df['pl_intensity']) * 0.35 +
        normalize_series(df['quantum_yield_percent']) * 0.25 +
        normalize_series(df['lifetime_ns']) * 0.15
    )
    
    return df


def train_cis_te_models(df, feature_cols, target_cols):
    """Train multiple models for CIS-Te/ZnS optimization"""
    
    # Prepare features
    X = df[feature_cols].copy()
    
    # Handle any missing values
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {}
    results = {}
    
    for target in target_cols:
        if target not in df.columns:
            continue
        
        y = df[target].values
        
        # Train multiple models
        best_r2 = -np.inf
        best_model = None
        best_name = None
        
        model_configs = [
            ('Random Forest', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
            ('Gradient Boosting', GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)),
            ('SVR', SVR(kernel='rbf', C=100, gamma=0.1)),
            ('Linear Regression', LinearRegression())
        ]
        
        for name, model in model_configs:
            try:
                model.fit(X_scaled, y)
                r2 = model.score(X_scaled, y)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
                    best_name = name
            except:
                continue
        
        if best_model:
            models[target] = best_model
            results[target] = {'r2': best_r2, 'model_name': best_name}
    
    return models, results, scaler


def bayesian_optimize_cis_te(models, scaler, feature_cols, target_weights=None):
    """Bayesian optimization for CIS-Te/ZnS synthesis"""
    
    if target_weights is None:
        target_weights = {'pl_intensity': 0.4, 'emission_nm': 0.3, 'quantum_yield_percent': 0.2, 'lifetime_ns': 0.1}
    
    # Define search space
    space = [
        Real(8.0, 14.0, name='cucl2_mg'),
        Real(40.0, 70.0, name='incl3_mg'),
        Real(1.0, 2.5, name='te_salt_mg'),
        Real(20.0, 40.0, name='zn_precursor_mg'),
        Real(85.0, 105.0, name='core_temp_c'),
        Integer(20, 40, name='core_time_min'),
        Real(2.5, 4.0, name='pH'),
        Integer(10, 30, name='shell_time_min')
    ]
    
    # Extract available features
    available_features = [f for f in feature_cols if f in [s.name for s in space]]
    
    def objective(params):
        param_dict = dict(zip([s.name for s in space], params))
        
        # Create input array
        X_new = np.array([[param_dict.get(f, 0) for f in available_features]])
        X_scaled = scaler.transform(X_new)
        
        score = 0
        for target, weight in target_weights.items():
            if target in models:
                pred = models[target].predict(X_scaled)[0]
                
                # Normalize predictions for scoring
                if target == 'emission_nm':
                    # Prefer emission near 815 nm
                    normalized = 1 - abs(pred - 815) / 100
                elif target == 'pl_intensity':
                    normalized = pred / 15000
                elif target == 'quantum_yield_percent':
                    normalized = pred / 10
                elif target == 'lifetime_ns':
                    normalized = pred / 50
                else:
                    normalized = pred / (pred + 1)
                
                score += weight * max(0, min(1, normalized))
        
        return -score  # Minimize negative score
    
    if SKOPT_AVAILABLE:
        try:
            result = gp_minimize(
                objective, space,
                n_calls=80,
                n_initial_points=20,
                random_state=42,
                acq_func='EI'
            )
            
            optimal_params = {}
            for i, s in enumerate(space):
                optimal_params[s.name] = result.x[i]
            
            return optimal_params, -result.fun
        except:
            return None, 0
    else:
        # Random search fallback
        best_score = -np.inf
        best_params = None
        
        for _ in range(100):
            params = [np.random.uniform(s.low, s.high) if isinstance(s, Real) 
                     else np.random.randint(s.low, s.high + 1) for s in space]
            score = -objective(params)
            
            if score > best_score:
                best_score = score
                best_params = {s.name: params[i] for i, s in enumerate(space)}
        
        return best_params, best_score


def multi_objective_optimization_cis_te(df, n_points=100):
    """Multi-objective Pareto optimization for CIS-Te/ZnS"""
    
    # Define objectives
    objectives = ['pl_intensity', 'quantum_yield_percent', 'lifetime_ns']
    
    # Normalize objectives
    df_normalized = df.copy()
    for obj in objectives:
        if obj in df.columns:
            df_normalized[obj + '_norm'] = normalize_series(df[obj])
    
    # Calculate Pareto front
    if all(obj + '_norm' in df_normalized.columns for obj in objectives):
        # For 2D Pareto (use intensity and QY)
        obj1 = df_normalized['pl_intensity_norm'].values
        obj2 = df_normalized['quantum_yield_percent_norm'].values
        
        is_pareto = np.ones(len(df), dtype=bool)
        for i in range(len(df)):
            for j in range(len(df)):
                if i != j:
                    if (obj1[j] >= obj1[i] and obj2[j] >= obj2[i] and
                        (obj1[j] > obj1[i] or obj2[j] > obj2[i])):
                        is_pareto[i] = False
                        break
        
        pareto_df = df[is_pareto].copy()
        pareto_df = pareto_df.sort_values('pl_intensity', ascending=False)
        
        return pareto_df
    else:
        # Return top performing experiments
        if 'composite_score' in df.columns:
            return df.sort_values('composite_score', ascending=False).head(n_points)
        else:
            return df.sort_values('pl_intensity', ascending=False).head(n_points)


def pls_regression_cis_te(df, feature_cols, target_cols):
    """Partial Least Squares Regression for CIS-Te/ZnS"""
    try:
        from sklearn.cross_decomposition import PLSRegression
        
        results = {}
        
        for target in target_cols:
            if target not in df.columns:
                continue
            
            X = df[feature_cols].values
            y = df[target].values
            
            # Find optimal number of components
            best_n_components = 2
            best_r2 = -np.inf
            
            for n in range(2, min(8, len(feature_cols))):
                pls = PLSRegression(n_components=n)
                pls.fit(X, y)
                r2 = pls.score(X, y)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_n_components = n
            
            pls = PLSRegression(n_components=best_n_components)
            pls.fit(X, y)
            
            # Get feature importance
            importance = np.abs(pls.coef_).flatten()
            importance_dict = {feature_cols[i]: importance[i] for i in range(len(feature_cols))}
            
            results[target] = {
                'model': pls,
                'r2': best_r2,
                'n_components': best_n_components,
                'feature_importance': importance_dict
            }
        
        return results
    except ImportError:
        return {}


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_cis_te_optimizer(data, qd_manager):
    """Specialized CIS-Te/ZnS Quantum Dot Optimizer with updated experimental parameters"""
    
    st.markdown("### 👨‍🔬 CIS-Te/ZnS Quantum Dot Optimizer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Parameter Ranges (Based on Dr. Adimula 2026)")
        
        # CIS-Te/ZnS specific parameters with experimental values
        cucl2 = st.slider("CuCl₂ (mg)", 8.0, 14.0, 11.0, 0.5, key="cite_cucl2",
                         help="Copper precursor - 11.0 mg used in optimized synthesis")
        incl3 = st.slider("InCl₃ (mg)", 40.0, 70.0, 55.0, 1.0, key="cite_incl3",
                         help="Indium precursor - 55.0 mg used in optimized synthesis")
        te_salt = st.slider("Tellurium Salt (mg)", 1.0, 2.5, 1.7, 0.1, key="cite_te",
                           help="Te precursor - 1.7 mg used (critical for red shift)")
        zn_precursor = st.slider("Zn(OAc)₂ (mg)", 20.0, 40.0, 30.0, 1.0, key="cite_zn",
                                help="Zinc precursor for shell - 30.0 mg used")
        core_temp = st.slider("Core Temperature (°C)", 85.0, 105.0, 95.0, 1.0, key="cite_core_temp",
                             help="Core formation temperature - 95°C used")
        core_time = st.slider("Core Time (min)", 20, 40, 30, 2, key="cite_core_time",
                             help="Core reaction time - 30 min used")
        pH_val = st.slider("pH", 2.5, 4.0, 3.0, 0.1, key="cite_pH",
                          help="pH adjusted to 3.0 with HCl")
        shell_time = st.slider("Shell Growth Time (min)", 10, 30, 20, 2, key="cite_shell_time",
                              help="Shell growth time - 20 min used")
        
        current_params = {
            "cucl2_mg": cucl2, "incl3_mg": incl3, "te_salt_mg": te_salt,
            "zn_precursor_mg": zn_precursor, "core_temp_c": core_temp,
            "core_time_min": core_time, "pH": pH_val, "shell_time_min": shell_time
        }
        
        # Store ranges for optimization
        ranges = {
            "cucl2_mg": (8.0, 14.0), "incl3_mg": (40.0, 70.0), "te_salt_mg": (1.0, 2.5),
            "zn_precursor_mg": (20.0, 40.0), "core_temp_c": (85.0, 105.0),
            "core_time_min": (20, 40), "pH": (2.5, 4.0), "shell_time_min": (10, 30)
        }
    
    with col2:
        st.markdown("#### Optimization Targets")
        
        st.markdown("**Experimental Best Values (Dr. Adimula, 2026):**")
        st.metric("Emission Wavelength", "815 nm", delta="from 827 nm (core)")
        st.metric("PL Intensity", "12,829", delta="+342% from core")
        st.metric("Quantum Yield", "5.13%", delta="+526% from core")
        st.metric("Lifetime", "37.54 ns", delta="+59% from core")
        
        st.markdown("---")
        st.markdown("**Target Optimization Settings:**")
        
        target_emission = st.number_input("Target Emission (nm)", 750, 900, 815, key="cite_target_emission")
        target_intensity = st.number_input("Target PL Intensity", 5000, 20000, 12829, key="cite_target_intensity")
        target_qy = st.number_input("Target Quantum Yield (%)", 2.0, 15.0, 5.13, 0.5, key="cite_target_qy")
        target_lifetime = st.number_input("Target Lifetime (ns)", 30, 60, 38, key="cite_target_lifetime")
        
        # Display Cu:In and Te ratios
        cu_in_ratio = cucl2 / incl3 if incl3 > 0 else 0
        st.metric("Cu:In Ratio", f"{cu_in_ratio:.3f}", 
                 help="Optimal ratio ~0.2 (based on 11.0/55.0 = 0.2)")
    
    # Load or generate CIS-Te/ZnS specific data
    st.markdown("---")
    st.markdown("#### Experimental Data")
    
    cite_uploaded = st.file_uploader("Upload CIS-Te/ZnS experimental CSV", type="csv", key="cite_upload")
    
    if cite_uploaded:
        try:
            cite_df = pd.read_csv(cite_uploaded)
            for col in cite_df.columns:
                if col in EXPERIMENTAL_PARAM_RANGES.keys():
                    cite_df[col] = pd.to_numeric(cite_df[col], errors='coerce')
            cite_df = cite_df.dropna()
            st.success(f"✅ Loaded {len(cite_df)} experimental data points")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            cite_df = generate_cis_te_data(n_samples=40)
    else:
        with st.spinner("Generating synthetic CIS-Te/ZnS data based on experimental results..."):
            cite_df = generate_cis_te_data(n_samples=40)
        st.info("📊 Using synthetic data based on experimental results. Upload your own CSV for real optimization.")
    
    st.dataframe(cite_df.head(10), use_container_width=True)
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Emission", f"{cite_df['emission_nm'].min():.1f} nm")
    with col2:
        st.metric("Best Intensity", f"{cite_df['pl_intensity'].max():.0f}")
    with col3:
        st.metric("Best QY", f"{cite_df['quantum_yield_percent'].max():.2f}%")
    with col4:
        st.metric("Best Lifetime", f"{cite_df['lifetime_ns'].max():.1f} ns")
    
    # Optimization section
    st.markdown("---")
    st.markdown("#### 🚀 Multi-Objective Optimization Tools")
    
    # Feature columns for modeling
    feature_cols = ['cucl2_mg', 'incl3_mg', 'te_salt_mg', 'zn_precursor_mg', 
                   'core_temp_c', 'core_time_min', 'pH', 'shell_time_min']
    
    available_features = [f for f in feature_cols if f in cite_df.columns]
    target_cols = ['emission_nm', 'pl_intensity', 'quantum_yield_percent', 'lifetime_ns']
    available_targets = [t for t in target_cols if t in cite_df.columns]
    
    if available_features and available_targets:
        # Train models
        models, model_results, scaler = train_cis_te_models(cite_df, available_features, available_targets)
        
        # Display model performance
        with st.expander("📊 Model Performance"):
            st.markdown("**Trained Models R² Scores:**")
            for target, result in model_results.items():
                st.metric(f"{target}", f"R² = {result['r2']:.3f}", f"Model: {result['model_name']}")
        
        # Optimization buttons
        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
        
        with opt_col1:
            if st.button("🎯 Bayesian Optimization", use_container_width=True):
                with st.spinner("Running Bayesian optimization on parameter space..."):
                    optimal_params, best_score = bayesian_optimize_cis_te(models, scaler, available_features)
                    
                    if optimal_params:
                        st.success("✅ Optimal synthesis conditions found:")
                        
                        for k, v in optimal_params.items():
                            st.metric(k.replace('_', ' ').title(), f"{v:.2f}")
                        
                        # Predict properties at optimal
                        X_opt = np.array([[optimal_params.get(f, 0) for f in available_features]])
                        X_opt_scaled = scaler.transform(X_opt)
                        
                        st.markdown("---")
                        st.markdown("**Predicted Properties:**")
                        
                        pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
                        with pred_col1:
                            if 'emission_nm' in models:
                                pred_em = models['emission_nm'].predict(X_opt_scaled)[0]
                                st.metric("Predicted Emission", f"{pred_em:.1f} nm")
                        with pred_col2:
                            if 'pl_intensity' in models:
                                pred_int = models['pl_intensity'].predict(X_opt_scaled)[0]
                                st.metric("Predicted Intensity", f"{pred_int:.0f}")
                        with pred_col3:
                            if 'quantum_yield_percent' in models:
                                pred_qy = models['quantum_yield_percent'].predict(X_opt_scaled)[0]
                                st.metric("Predicted QY", f"{pred_qy:.2f}%")
                        with pred_col4:
                            if 'lifetime_ns' in models:
                                pred_life = models['lifetime_ns'].predict(X_opt_scaled)[0]
                                st.metric("Predicted Lifetime", f"{pred_life:.1f} ns")
                    else:
                        st.error("Optimization failed. Try training models first.")
        
        with opt_col2:
            if st.button("📊 Pareto Front Analysis", use_container_width=True):
                with st.spinner("Calculating Pareto front..."):
                    pareto_df = multi_objective_optimization_cis_te(cite_df)
                    
                    st.success(f"✅ Found {len(pareto_df)} Pareto-optimal solutions")
                    
                    # Plot Pareto front
                    if 'pl_intensity' in pareto_df.columns and 'quantum_yield_percent' in pareto_df.columns:
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=cite_df['pl_intensity'],
                            y=cite_df['quantum_yield_percent'],
                            mode='markers',
                            name='All Experiments',
                            marker=dict(color='lightblue', size=6, opacity=0.5)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=pareto_df['pl_intensity'],
                            y=pareto_df['quantum_yield_percent'],
                            mode='markers+lines',
                            name='Pareto Front',
                            marker=dict(color='red', size=10, symbol='star'),
                            line=dict(dash='dash', color='red')
                        ))
                        
                        fig.update_layout(
                            title="Pareto Front: PL Intensity vs Quantum Yield",
                            xaxis_title="PL Intensity (a.u.)",
                            yaxis_title="Quantum Yield (%)",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display top Pareto solutions
                        st.markdown("**Top Pareto-Optimal Solutions:**")
                        display_cols = [c for c in ['cucl2_mg', 'incl3_mg', 'te_salt_mg', 'zn_precursor_mg',
                                                    'core_temp_c', 'core_time_min', 'pH', 'shell_time_min',
                                                    'pl_intensity', 'quantum_yield_percent', 'emission_nm', 'lifetime_ns']
                                        if c in pareto_df.columns]
                        st.dataframe(pareto_df[display_cols].head(10), use_container_width=True)
        
        with opt_col3:
            if st.button("🔮 Predict Current Conditions", use_container_width=True):
                with st.spinner("Predicting properties..."):
                    X_current = np.array([[current_params.get(f, 0) for f in available_features]])
                    X_scaled = scaler.transform(X_current)
                    
                    st.markdown("**Predicted Properties for Current Conditions:**")
                    
                    pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
                    
                    with pred_col1:
                        if 'emission_nm' in models:
                            pred_em = models['emission_nm'].predict(X_scaled)[0]
                            delta_em = pred_em - EXPERIMENTAL_OPTIMUM['emission_nm']
                            st.metric("Emission", f"{pred_em:.1f} nm", 
                                    delta=f"{delta_em:+.1f} nm", 
                                    delta_color="inverse")
                    
                    with pred_col2:
                        if 'pl_intensity' in models:
                            pred_int = models['pl_intensity'].predict(X_scaled)[0]
                            delta_int = pred_int - EXPERIMENTAL_OPTIMUM['pl_intensity']
                            st.metric("PL Intensity", f"{pred_int:.0f}", 
                                    delta=f"{delta_int:+.0f}")
                    
                    with pred_col3:
                        if 'quantum_yield_percent' in models:
                            pred_qy = models['quantum_yield_percent'].predict(X_scaled)[0]
                            delta_qy = pred_qy - EXPERIMENTAL_OPTIMUM['quantum_yield_percent']
                            st.metric("QY", f"{pred_qy:.2f}%", 
                                    delta=f"{delta_qy:+.2f}%")
                    
                    with pred_col4:
                        if 'lifetime_ns' in models:
                            pred_life = models['lifetime_ns'].predict(X_scaled)[0]
                            delta_life = pred_life - EXPERIMENTAL_OPTIMUM['lifetime_ns']
                            st.metric("Lifetime", f"{pred_life:.1f} ns", 
                                    delta=f"{delta_life:+.1f} ns")
        
        with opt_col4:
            if st.button("📈 Sensitivity Analysis", use_container_width=True):
                with st.spinner("Analyzing parameter sensitivity..."):
                    sensitivity_results = {}
                    
                    for param in available_features:
                        param_range = EXPERIMENTAL_PARAM_RANGES.get(param, (0, 100))
                        test_values = np.linspace(param_range[0], param_range[1], 20)
                        
                        sensitivity = {}
                        for target in available_targets:
                            if target in models:
                                predictions = []
                                base_params = {p: current_params.get(p, 0) for p in available_features}
                                
                                for val in test_values:
                                    test_params = base_params.copy()
                                    test_params[param] = val
                                    X_test = np.array([[test_params.get(f, 0) for f in available_features]])
                                    X_test_scaled = scaler.transform(X_test)
                                    pred = models[target].predict(X_test_scaled)[0]
                                    predictions.append(pred)
                                
                                sensitivity[target] = np.std(predictions) / np.mean(predictions) if np.mean(predictions) > 0 else 0
                        
                        sensitivity_results[param] = sensitivity
                    
                    # Create sensitivity plot
                    fig = go.Figure()
                    
                    for target in available_targets:
                        sensitivities = [sensitivity_results[p].get(target, 0) for p in available_features]
                        fig.add_trace(go.Bar(
                            x=available_features,
                            y=sensitivities,
                            name=target,
                            text=[f"{s:.3f}" for s in sensitivities],
                            textposition='auto'
                        ))
                    
                    fig.update_layout(
                        title="Parameter Sensitivity Analysis",
                        xaxis_title="Parameter",
                        yaxis_title="Sensitivity (CV)",
                        barmode='group',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("Higher bars indicate greater influence on the target property.")
        
        # PLS Regression
        st.markdown("---")
        st.markdown("#### 📐 Partial Least Squares (PLS) Regression Analysis")
        
        if st.button("Run PLS Regression Analysis", use_container_width=True):
            with st.spinner("Running PLS regression analysis..."):
                pls_results = pls_regression_cis_te(cite_df, available_features, available_targets)
                
                if pls_results:
                    for target, result in pls_results.items():
                        st.markdown(f"**{target} - PLS Regression (R² = {result['r2']:.3f}, n_components = {result['n_components']})**")
                        
                        # Feature importance
                        importance_df = pd.DataFrame([
                            {'Parameter': p, 'Importance': result['feature_importance'].get(p, 0)}
                            for p in available_features
                        ]).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(importance_df, x='Parameter', y='Importance', 
                                    title=f"Feature Importance for {target}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("PLS regression requires scikit-learn with cross_decomposition module.")
    
    return cite_df


def display_quantum_dots_tab(uploaded_file):
    """Quantum Dots tab with advanced ML/AI subtabs including CIS-Te/ZnS Optimizer"""
    
    st.markdown("<h2 class='sub-header'>🧪 Quantum Dot Synthesis Optimization Suite</h2>", unsafe_allow_html=True)
    
    # QD Type Selection
    qd_type = st.selectbox(
        "Select Quantum Dot Type",
        ["CIS/ZnS", "CIS-Te/ZnS", "AIS/ZnS", "CdSe/CdS", "Carbon Dots", "Metal Nanoparticles"],
        key="qd_type"
    )
    
    # Initialize QD Data Manager
    qd_manager = QDDataManager()
    
    # Display QD information
    with st.expander(f"ℹ️ About {qd_type}", expanded=False):
        if qd_type in qd_manager.qd_types:
            info = qd_manager.qd_types[qd_type]
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Optimal Absorption:** {info.get('optimal_absorption', 'N/A')}")
            st.markdown(f"**Optimal PLQY:** {info.get('optimal_plqy', 'N/A')}")
            st.markdown(f"**Key Parameters:** {', '.join(info['key_params'])}")
            
            # Show additional metrics for CIS-Te/ZnS
            if qd_type == "CIS-Te/ZnS":
                st.markdown(f"**Optimal Emission:** {info.get('optimal_emission', '815 nm')}")
                st.markdown(f"**Optimal Intensity:** {info.get('optimal_intensity', '12,829 a.u.')}")
                st.markdown(f"**Optimal Lifetime:** {info.get('optimal_lifetime', '37.54 ns')}")
        else:
            st.markdown(f"**Description:** Copper Indium Sulfide / Zinc Sulfide quantum dots")
            st.markdown(f"**Optimal Absorption:** 650-850 nm")
            st.markdown(f"**Optimal PLQY:** 50-80%")
    
    # Load or generate data
    if uploaded_file is not None:
        data = DataManager.load_data(uploaded_file)
        if data is None:
            st.warning("⚠️ Could not load uploaded file. Using generated data instead.")
            data = generate_cis_te_data() if qd_type == "CIS-Te/ZnS" else qd_manager.generate_sample_data(qd_type, 100)
    else:
        if qd_type == "CIS-Te/ZnS":
            data = generate_cis_te_data()
            st.info("📊 Using generated CIS-Te/ZnS sample data based on Dr. Adimula (2026) results.")
        else:
            data = qd_manager.generate_sample_data(qd_type, 100)
            st.info(f"📊 Using generated {qd_type} sample data. Upload your own CSV for real optimization.")
    
    if data is None or len(data) == 0:
        st.error("No data available")
        return
    
    # Create enhanced tabs with CIS-Te/ZnS Optimizer
    qd_tabs = st.tabs([
        "📊 Data Explorer",
        "👨‍🔬 CIS-Te/ZnS Optimizer",
        "🔮 Molecular & Optical Properties",
        "📐 Design of Experiments",
        "🤖 Reinforcement Learning",
        "📈 Supervised Learning",
        "🔬 Optimization",
        "📥 Export"
    ])
    
    # Tab 1: Data Explorer
    with qd_tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Experimental Data")
            st.dataframe(data.head(10), use_container_width=True)
            
            with st.expander("📊 Summary Statistics"):
                st.dataframe(data.describe(), use_container_width=True)
        
        with col2:
            st.markdown("### Data Overview")
            st.metric("Total Experiments", len(data))
            st.metric("Features", len(data.columns))
            
            st.markdown("### 🎯 Target Properties")
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Define potential target properties for CIS-Te/ZnS
            if qd_type == "CIS-Te/ZnS":
                potential_targets = ['emission_nm', 'pl_intensity', 'quantum_yield_percent', 'lifetime_ns', 
                                    'absorption_nm', 'excitation_nm', 'composite_score']
            else:
                potential_targets = ['absorption_nm', 'plqy_percent', 'fwhm_nm', 'quantum_yield', 'size_nm']
            
            available_targets = [t for t in potential_targets if t in numeric_cols]
            
            if not available_targets and len(numeric_cols) > 0:
                available_targets = numeric_cols[:4]
            
            for target in available_targets[:4]:
                col_a, col_b = st.columns(2)
                with col_a:
                    try:
                        if target == 'emission_nm':
                            best_val = data[target].min()
                            st.metric(f"Best {target}", f"{best_val:.1f} nm")
                        elif target == 'pl_intensity':
                            best_val = data[target].max()
                            st.metric(f"Best {target}", f"{best_val:.0f}")
                        elif target == 'quantum_yield_percent':
                            best_val = data[target].max()
                            st.metric(f"Best {target}", f"{best_val:.2f}%")
                        elif target == 'lifetime_ns':
                            best_val = data[target].max()
                            st.metric(f"Best {target}", f"{best_val:.1f} ns")
                        else:
                            best_val = data[target].max()
                            if pd.api.types.is_numeric_dtype(data[target]):
                                st.metric(f"Best {target}", f"{best_val:.2f}")
                            else:
                                st.metric(f"Best {target}", str(best_val))
                    except Exception as e:
                        st.metric(f"Best {target}", "N/A")
                
                with col_b:
                    try:
                        mean_val = data[target].mean()
                        if pd.api.types.is_numeric_dtype(data[target]):
                            st.metric(f"Mean {target}", f"{mean_val:.2f}")
                        else:
                            st.metric(f"Mean {target}", str(mean_val))
                    except Exception as e:
                        st.metric(f"Mean {target}", "N/A")
    
    # Tab 2: CIS-Te/ZnS Optimizer
    with qd_tabs[1]:
        data = display_cis_te_optimizer(data, qd_manager)
    
    # Tab 3: Molecular & Optical Properties
    with qd_tabs[2]:
        st.markdown("### 🔮 Molecular and Optical Property Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Input Parameters")
            
            inputs = {}
            if qd_type == "CIS-Te/ZnS":
                params = ['cucl2_mg', 'incl3_mg', 'te_salt_mg', 'zn_precursor_mg', 
                         'core_temp_c', 'core_time_min', 'pH', 'shell_time_min']
            else:
                params = qd_manager.qd_types.get(qd_type, {'key_params': []})['key_params']
            
            for param in params:
                if param in data.columns and pd.api.types.is_numeric_dtype(data[param]):
                    min_val = float(data[param].min())
                    max_val = float(data[param].max())
                    default_val = float(data[param].mean())
                    
                    inputs[param] = st.slider(
                        f"{param.replace('_', ' ').title()}",
                        min_val, max_val, default_val,
                        key=f"input_{param}"
                    )
        
        with col2:
            st.markdown("#### Predicted Properties")
            
            if st.button("🔮 Predict Properties", use_container_width=True):
                with st.spinner("Training prediction models..."):
                    # Prepare features and targets based on QD type
                    if qd_type == "CIS-Te/ZnS":
                        feature_cols = ['cucl2_mg', 'incl3_mg', 'te_salt_mg', 'zn_precursor_mg', 
                                       'core_temp_c', 'core_time_min', 'pH', 'shell_time_min']
                        target_cols = ['emission_nm', 'pl_intensity', 'quantum_yield_percent', 'lifetime_ns']
                    else:
                        feature_cols = [p for p in qd_manager.qd_types.get(qd_type, {'key_params': []})['key_params'] 
                                      if p in data.columns and pd.api.types.is_numeric_dtype(data[p])]
                        target_cols = ['absorption_nm', 'plqy_percent', 'fwhm_nm', 'quantum_yield']
                    
                    # Check available columns
                    available_features = [f for f in feature_cols if f in data.columns]
                    available_targets = [t for t in target_cols if t in data.columns]
                    
                    if len(available_features) < 2:
                        st.error("Not enough numeric feature columns for prediction")
                    elif len(available_targets) == 0:
                        st.error("No numeric target columns found")
                    else:
                        # Train models
                        models, results, scaler = train_cis_te_models(data, available_features, available_targets)
                        
                        # Prepare input
                        X_input = np.array([[inputs.get(f, 0) for f in available_features]])
                        X_scaled = scaler.transform(X_input)
                        
                        # Display predictions
                        for target in available_targets:
                            if target in models:
                                pred = models[target].predict(X_scaled)[0]
                                r2 = results.get(target, {}).get('r2', 0)
                                
                                if target == 'emission_nm':
                                    st.metric("Predicted Emission", f"{pred:.1f} nm", f"R²: {r2:.3f}")
                                elif target == 'pl_intensity':
                                    st.metric("Predicted PL Intensity", f"{pred:.0f}", f"R²: {r2:.3f}")
                                elif target == 'quantum_yield_percent':
                                    st.metric("Predicted Quantum Yield", f"{pred:.2f}%", f"R²: {r2:.3f}")
                                elif target == 'lifetime_ns':
                                    st.metric("Predicted Lifetime", f"{pred:.1f} ns", f"R²: {r2:.3f}")
                                else:
                                    st.metric(f"Predicted {target}", f"{pred:.2f}", f"R²: {r2:.3f}")
    
    # Tab 4: Design of Experiments
    with qd_tabs[3]:
        st.markdown("### 📐 Design of Experiments for QD Synthesis")
        
        st.markdown("""
        <div class='info-box'>
        Design optimal experiments to explore the parameter space efficiently.
        For CIS-Te/ZnS, focus on CuCl₂, InCl₃, Te salt, and Zn precursor ratios.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Factor Ranges")
            
            # Get factor ranges from experimental data
            if qd_type == "CIS-Te/ZnS":
                factors = EXPERIMENTAL_PARAM_RANGES.copy()
            else:
                factors = {}
                param_list = qd_manager.qd_types.get(qd_type, {'key_params': []})['key_params']
                
                for param in param_list:
                    if param in data.columns and pd.api.types.is_numeric_dtype(data[param]):
                        min_val = float(data[param].min())
                        max_val = float(data[param].max())
                        range_vals = st.slider(
                            f"{param.replace('_', ' ').title()} Range",
                            min_val, max_val, (min_val, max_val),
                            key=f"range_{param}"
                        )
                        factors[param] = range_vals
            
            if len(factors) == 0:
                st.warning("No numeric parameters available for DoE")
        
        with col2:
            st.markdown("#### Design Parameters")
            
            if len(factors) > 0:
                design_type = st.selectbox(
                    "Design Type",
                    ["Full Factorial", "Fractional Factorial", "Latin Hypercube", "Central Composite"],
                    key="doe_type"
                )
                
                n_experiments = st.number_input("Number of Experiments", 8, 100, 30, key="doe_n") if design_type == "Latin Hypercube" else None
                
                if st.button("🎲 Generate Experimental Design", use_container_width=True):
                    with st.spinner("Generating design..."):
                        # Generate Latin Hypercube design
                        factor_names = list(factors.keys())
                        n_factors = len(factor_names)
                        
                        sampler = qmc.LatinHypercube(d=n_factors)
                        sample = sampler.random(n=n_experiments) if n_experiments else sampler.random(n=30)
                        
                        design_data = {}
                        for i, name in enumerate(factor_names):
                            low, high = factors[name]
                            design_data[name] = qmc.scale(sample[:, i], low, high)
                        
                        design = pd.DataFrame(design_data)
                        design['run_order'] = np.random.permutation(len(design)) + 1
                        
                        st.success(f"✅ Generated {len(design)} experimental runs")
                        st.dataframe(design, use_container_width=True)
                        
                        # Download design
                        csv = design.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Design as CSV",
                            data=csv,
                            file_name=f"qd_doe_{qd_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Visualize design space
                        if n_factors >= 2:
                            fig = px.scatter_matrix(
                                design[factor_names[:min(4, n_factors)]],
                                title=f"{design_type} Design Space"
                            )
                            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Optimization
    with qd_tabs[6]:
        st.markdown("### 🔬 Multi-Objective Optimization")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for optimization")
        else:
            # For CIS-Te/ZnS, suggest appropriate objectives
            if qd_type == "CIS-Te/ZnS":
                default_objectives = ['pl_intensity', 'quantum_yield_percent', 'emission_nm', 'lifetime_ns']
                available_objectives = [o for o in default_objectives if o in numeric_cols]
            else:
                available_objectives = numeric_cols
            
            col1, col2 = st.columns(2)
            
            with col1:
                obj1 = st.selectbox("Primary Objective", available_objectives, key="opt_obj1")
            
            with col2:
                obj2 = st.selectbox("Secondary Objective", [o for o in available_objectives if o != obj1], key="opt_obj2")
            
            optimization_method = st.selectbox(
                "Optimization Method",
                ["Bayesian Optimization", "Grid Search", "Pareto Front Optimization"],
                key="opt_method"
            )
            
            n_iterations = st.number_input("Number of Iterations", 5, 100, 30, key="opt_iter")
            
            if st.button("🚀 Run Optimization", use_container_width=True):
                with st.spinner("Running optimization..."):
                    progress_bar = st.progress(0)
                    
                    if optimization_method == "Pareto Front Optimization":
                        pareto_df = multi_objective_optimization_cis_te(data, n_points=20)
                        
                        st.success(f"✅ Found {len(pareto_df)} Pareto-optimal solutions")
                        
                        # Plot Pareto front
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=data[obj1],
                            y=data[obj2],
                            mode='markers',
                            name='All Experiments',
                            marker=dict(color='lightblue', size=8, opacity=0.5)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=pareto_df[obj1],
                            y=pareto_df[obj2],
                            mode='markers+lines',
                            name='Pareto Front',
                            marker=dict(color='red', size=12, symbol='star'),
                            line=dict(dash='dash', color='red')
                        ))
                        
                        fig.update_layout(
                            title=f"Pareto Front: {obj1} vs {obj2}",
                            xaxis_title=obj1,
                            yaxis_title=obj2,
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display top solutions
                        st.markdown("**Top Pareto-Optimal Solutions:**")
                        display_cols = [obj1, obj2] + [c for c in ['cucl2_mg', 'incl3_mg', 'te_salt_mg', 'zn_precursor_mg',
                                                                   'core_temp_c', 'core_time_min', 'pH', 'shell_time_min']
                                                        if c in pareto_df.columns]
                        st.dataframe(pareto_df[display_cols].head(10), use_container_width=True)
                    
                    elif optimization_method == "Bayesian Optimization":
                        # Train a composite model
                        feature_cols = [c for c in EXPERIMENTAL_PARAM_RANGES.keys() if c in data.columns]
                        
                        if 'composite_score' in data.columns:
                            target = 'composite_score'
                        else:
                            # Create composite score
                            weights = {'pl_intensity': 0.35, 'quantum_yield_percent': 0.35, 
                                      'emission_nm': 0.15, 'lifetime_ns': 0.15}
                            data['composite_score'] = calculate_composite_score(data, weights)
                            target = 'composite_score'
                        
                        if feature_cols and target in data.columns:
                            X = data[feature_cols].values
                            y = data[target].values
                            
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(X, y)
                            
                            # Define search space
                            space = []
                            for col in feature_cols:
                                low, high = EXPERIMENTAL_PARAM_RANGES.get(col, (data[col].min(), data[col].max()))
                                space.append(Real(low, high, name=col))
                            
                            def objective(params):
                                X_test = np.array([params])
                                return -model.predict(X_test)[0]
                            
                            if SKOPT_AVAILABLE:
                                result = gp_minimize(objective, space, n_calls=n_iterations, 
                                                   n_initial_points=10, random_state=42)
                                
                                optimal_params = {space[i].name: result.x[i] for i in range(len(space))}
                                
                                st.success("✅ Optimal conditions found:")
                                for param, value in optimal_params.items():
                                    st.metric(param.replace('_', ' ').title(), f"{value:.2f}")
                                
                                # Predict optimal score
                                X_opt = np.array([result.x])
                                opt_score = model.predict(X_opt)[0]
                                st.metric("Predicted Composite Score", f"{opt_score:.3f}")
                            else:
                                # Random search fallback
                                best_score = -np.inf
                                best_params = None
                                
                                for i in range(n_iterations):
                                    progress_bar.progress((i + 1) / n_iterations)
                                    params = [np.random.uniform(s.low, s.high) for s in space]
                                    score = -objective(params)
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_params = {space[j].name: params[j] for j in range(len(space))}
                                
                                st.success("✅ Optimal conditions found (random search):")
                                for param, value in best_params.items():
                                    st.metric(param.replace('_', ' ').title(), f"{value:.2f}")
                        else:
                            st.error("Insufficient data for Bayesian optimization")
                    
                    else:  # Grid search
                        # Simple grid search over limited parameters
                        param_names = ['te_salt_mg', 'zn_precursor_mg', 'pH', 'shell_time_min']
                        available_grid_params = [p for p in param_names if p in data.columns]
                        
                        if available_grid_params:
                            st.info(f"Grid searching over: {available_grid_params}")
                            
                            best_score = -np.inf
                            best_combo = None
                            
                            n_grid_points = min(5, int(n_iterations ** (1/len(available_grid_params))))
                            
                            # Generate grid
                            grids = []
                            for param in available_grid_params:
                                low, high = EXPERIMENTAL_PARAM_RANGES.get(param, (data[param].min(), data[param].max()))
                                grids.append(np.linspace(low, high, n_grid_points))
                            
                            grid_combinations = list(product(*grids))
                            
                            for i, combo in enumerate(grid_combinations[:n_iterations]):
                                progress_bar.progress((i + 1) / min(len(grid_combinations), n_iterations))
                                
                                # Estimate composite score based on proximity to optimum
                                score = 0
                                for j, param in enumerate(available_grid_params):
                                    opt_val = EXPERIMENTAL_OPTIMUM.get(param, 0)
                                    if opt_val > 0:
                                        score += 1 - min(1, abs(combo[j] - opt_val) / opt_val)
                                
                                if score > best_score:
                                    best_score = score
                                    best_combo = combo
                            
                            if best_combo:
                                st.success("✅ Best grid search combination:")
                                for j, param in enumerate(available_grid_params):
                                    st.metric(param.replace('_', ' ').title(), f"{best_combo[j]:.2f}")
                        else:
                            st.warning("No parameters available for grid search")
    
    # Tab 8: Export
    with qd_tabs[7]:
        st.markdown("### 📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export to CSV", use_container_width=True):
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"qd_{qd_type}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Export Optimization Report", use_container_width=True):
                report = f"""# {qd_type} Quantum Dot Synthesis Optimization Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
Total Experiments: {len(data)}
Features: {len(data.columns)}

## Target Properties
"""
                if qd_type == "CIS-Te/ZnS":
                    report += f"""
Best Emission: {data['emission_nm'].min():.1f} nm
Best PL Intensity: {data['pl_intensity'].max():.0f}
Best Quantum Yield: {data['quantum_yield_percent'].max():.2f}%
Best Lifetime: {data['lifetime_ns'].max():.1f} ns

## Optimal Parameters (Experimental)
CuCl₂: 11.0 mg
InCl₃: 55.0 mg
Te Salt: 1.7 mg
Zn(OAc)₂: 30.0 mg
Core Temperature: 95°C
Core Time: 30 min
pH: 3.0
Shell Time: 20 min
"""
                else:
                    for col in data.select_dtypes(include=[np.number]).columns[:5]:
                        report += f"\n{col}: {data[col].mean():.2f} ± {data[col].std():.2f}"
                
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"qd_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )


# ============================================================================
# Placeholder functions for other tabs (to maintain compatibility)
# ============================================================================

def display_porphyrins_tab(uploaded_file):
    """Porphyrins tab content"""
    st.markdown("<h2 class='sub-header'>🔬 Porphyrin Synthesis Optimization</h2>", unsafe_allow_html=True)
    st.info("Porphyrin synthesis optimization tools - Upload your data to get started.")
    
    if uploaded_file is not None:
        data = DataManager.load_data(uploaded_file)
        if data is not None:
            st.dataframe(data.head(10), use_container_width=True)

def display_multi_objective_tab():
    """Multi-objective optimization tab"""
    st.markdown("<h2 class='sub-header'>🎯 Multi-Objective Pareto Optimization</h2>", unsafe_allow_html=True)
    st.info("Multi-objective optimization tools for balancing multiple synthesis targets.")

def display_molecular_generator_tab():
    """Molecular generator tab"""
    st.markdown("<h2 class='sub-header'>🧬 Molecular Generator</h2>", unsafe_allow_html=True)
    st.info("AI-powered molecular design tools - Coming soon with full RDKit integration.")

def display_advanced_visualization(uploaded_file):
    """Advanced visualization tab"""
    st.markdown("<h2 class='sub-header'>📊 Advanced Visualization</h2>", unsafe_allow_html=True)
    st.info("Advanced spectral analysis and visualization tools.")

def display_pce_tab():
    """Photothermal conversion efficiency tab"""
    st.markdown("<h2 class='sub-header'>🔥 Photothermal Conversion Efficiency Analyzer</h2>", unsafe_allow_html=True)
    st.info("PCE calculation with Q_dis correction for solvent effects.")

def display_ai_assistant():
    """AI assistant tab"""
    st.markdown("<h2 class='sub-header'>🤖 AI Research Assistant</h2>", unsafe_allow_html=True)
    st.info("AI-powered research assistant for synthesis optimization.")


# ============================================================================
# Main function
# ============================================================================

def main():
    try:
        with st.sidebar:
            # Display logo
            if os.path.exists("images") and os.listdir("images"):
                st.image(os.path.join("images", os.listdir("images")[0]), use_container_width=True)
            else:
                st.markdown("""
                <div class='sidebar-logo'>
                    <div style='font-size:3rem;'>🧪</div>
                    <div class='sidebar-logo-text'>CHEM‑NANO‑BEW</div>
                    <div style='color:#ecf0f1; font-size:0.9rem;'>LABORATORY</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Mode selection
            mode = st.radio("Select Mode", [
                "🧪 Quantum Dots",
                "🔬 Porphyrins", 
                "🎯 Multi-Objective",
                "🧬 Molecular Generator",
                "📊 Advanced Visualization",
                "🔥 PCE Analyzer",
                "🤖 AI Research Assistant"
            ])
            
            st.markdown("---")
            
            # Logo upload
            with st.expander("📸 Upload Logo"):
                logo = st.file_uploader("Choose image", type=['png','jpg','jpeg','gif'], key="logo_uploader")
                if logo is not None:
                    saved_path = save_uploaded_image(logo)
                    if saved_path:
                        st.success("✅ Logo uploaded!")
                        st.rerun()
            
            st.markdown("---")
            
            # Data upload
            st.markdown("## 📁 Data Management")
            uploaded_file = st.file_uploader("Upload CSV data", type=['csv'], key="data_uploader")
            
            if uploaded_file is not None:
                st.success(f"✅ Loaded: {uploaded_file.name}")
            
            st.markdown("---")
            
            # Experimental results summary
            with st.expander("📊 Experimental Results (Dr. Adimula, 2026)"):
                st.markdown("**Optimized CIS-Te/ZnS QDs:**")
                st.metric("Emission", "815 nm")
                st.metric("PL Intensity", "12,829 a.u.")
                st.metric("Quantum Yield", "5.13%")
                st.metric("Lifetime", "37.54 ns")
            
            st.markdown("---")
            
            # About section
            st.markdown("## ℹ️ About")
            st.info(
                "**CHEM‑NANO‑BEW Laboratory**\n\n"
                "Advanced synthesis optimization for "
                "quantum dots and porphyrins using "
                "machine learning and DoE.\n\n"
                "**Version:** 3.0 (CIS-Te/ZnS Update)\n\n"
                "Based on experimental results from Dr. Adimula (2026)"
            )
        
        # Main header
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<h1 class='main-header'>CHEM‑NANO‑BEW LABORATORY</h1>", unsafe_allow_html=True)
            st.markdown("<p class='lab-subtitle'>Advanced Synthesis Optimization Suite | CIS-Te/ZnS Optimizer Included</p>", unsafe_allow_html=True)

        # Route to appropriate tab
        if mode == "🧪 Quantum Dots":
            display_quantum_dots_tab(uploaded_file)
        elif mode == "🔬 Porphyrins":
            display_porphyrins_tab(uploaded_file)
        elif mode == "🎯 Multi-Objective":
            display_multi_objective_tab()
        elif mode == "🧬 Molecular Generator":
            display_molecular_generator_tab()
        elif mode == "📊 Advanced Visualization":
            display_advanced_visualization(uploaded_file)
        elif mode == "🔥 PCE Analyzer":
            display_pce_tab()
        elif mode == "🤖 AI Research Assistant":
            display_ai_assistant()
        else:
            st.error(f"Unknown mode selected: {mode}")
            
    except Exception as e:
        st.error(f"⚠️ An error occurred:")
        st.exception(e)
        st.info("Please check the console logs or refresh the page.")

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
            <p>Powered by CHEMNANOBEW GROUP • v3.0 (CIS-Te/ZnS Update)</p>
            <p style='font-size: 0.8rem;'>Based on experimental results from Dr. Adimula (2026)</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.set_page_config(page_title="CHEMNANOBEW - Error", page_icon="🚨")
        st.error("🚨 **Critical Application Error**")
        st.exception(e)
        st.markdown("""
        ### Troubleshooting Steps:
        1. Check that all required dependencies are installed
        2. Verify your API keys in `.streamlit/secrets.toml`
        3. Check the console for detailed error messages
        4. Try refreshing the page
        """)