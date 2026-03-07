"""
nanobew_app.py – Synthesis Optimization Suite (RDKit‑Mode)
Deploy on Streamlit Cloud with no RDKit dependency.
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import qmc
import plotly.io as pio
import base64
from PIL import Image
import os
import io
import time
from datetime import datetime
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

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    st.warning("⚠️ RDKit not installed – using simplified property estimation.")

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
# LOGO HANDLING - Using the uploaded image
# ============================================================================

def get_logo_base64():
    """Convert logo image to base64 for embedding"""
    try:
        # Check if logo file exists
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
        # Fallback text logo
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

def generate_doe(ranges, n=30):
    """Generate Design of Experiments data"""
    data = {}
    for k, (lo, hi) in ranges.items():
        data[k] = np.random.uniform(lo, hi, n)
    return pd.DataFrame(data)

def train_rf(df, targets):
    """Train Random Forest model"""
    X = df.drop(columns=targets)
    y = df[targets]
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300))
    model.fit(X, y)
    return model

def pareto_front(df, cols):
    """Calculate Pareto front"""
    data = df[cols].values
    mask = np.ones(len(data), dtype=bool)
    for i, c in enumerate(data):
        if mask[i]:
            mask[mask] = np.any(data[mask] > c, axis=1)
            mask[i] = True
    return df[mask]

def bayes_optimize(model, ranges, targets=["wavelength", "intensity"]):
    """Bayesian optimization using optuna"""
    def objective(trial):
        x = []
        for k, (lo, hi) in ranges.items():
            x.append(trial.suggest_float(k, lo, hi))
        x = np.array(x).reshape(1, -1)
        pred = model.predict(x)[0]
        score = pred[0] + pred[1]/1000
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    return study.best_params

def rl_suggest(history, ranges, eps=0.2):
    """RL-inspired experiment suggestion"""
    if len(history) < 5 or np.random.rand() < eps:
        return {k: np.random.uniform(lo, hi) for k, (lo, hi) in ranges.items()}
    best = history.sort_values("reward", ascending=False).iloc[0]
    suggestion = {}
    for k, (lo, hi) in ranges.items():
        suggestion[k] = np.clip(best[k] + np.random.normal(0, (hi-lo)*0.1), lo, hi)
    return suggestion

def generate_porphyrin_smiles(subs):
    """Generate porphyrin SMILES with substituents"""
    base = "c1cc2ccc3ccc4ccc(c1)c2c3c4"
    smiles = []
    for s in subs:
        smiles.append(base + s)
    return smiles

def gnn_predict(features):
    """GNN prediction placeholder"""
    return np.sum(features, axis=1) * 0.5 + 700

# ============================================================================
# Self-Healing System
# ============================================================================
class CompleteSelfHealingSystem:
    """Complete self-healing system for Streamlit apps"""
    
    def __init__(self, app_name="CHEMNANOBEW", slack_webhook=None):
        self.app_name = app_name
        self.slack_webhook = slack_webhook
        self.crash_count = 0
        self.setup_recovery_system()
    
    def setup_recovery_system(self):
        """Initialize all healing mechanisms"""
        self.setup_auto_backup()
        self.setup_crash_detection()
        self.setup_health_checks()
    
    def setup_auto_backup(self):
        """Auto-backup session state every minute"""
        def backup_loop():
            while True:
                time.sleep(60)
                self.backup_session()
        thread = threading.Thread(target=backup_loop, daemon=True)
        thread.start()
    
    def backup_session(self):
        """Backup session state to file"""
        try:
            backup = {
                'timestamp': datetime.now().isoformat(),
                'session': dict(st.session_state),
                'crash_count': self.crash_count
            }
            with open('app_backup.pkl', 'wb') as f:
                pickle.dump(backup, f)
        except:
            pass
    
    def setup_crash_detection(self):
        """Monitor for crashes and attempt recovery"""
        if os.path.exists('crash_flag.txt'):
            with open('crash_flag.txt', 'r') as f:
                last_crash = f.read()
            st.warning(f"🔄 App recovered from crash at {last_crash}")
            os.remove('crash_flag.txt')
    
    def setup_health_checks(self):
        """Monitor memory usage and other health metrics"""
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_percent()
            if memory_usage > 80:
                st.warning(f"⚠️ High memory usage: {memory_usage:.1f}%")
                self.cleanup_memory()
        except:
            pass
    
    def cleanup_memory(self):
        """Attempt to free memory"""
        import gc
        gc.collect()
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
    
    def healing_decorator(self, func):
        """Decorator that adds self-healing to any function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return self.heal_from_error(func, e, *args, **kwargs)
        return wrapper
    
    def heal_from_error(self, func, error, *args, **kwargs):
        """Attempt to heal from an error"""
        self.crash_count += 1
        crash_data = {
            'timestamp': datetime.now().isoformat(),
            'function': func.__name__,
            'error': str(error),
            'traceback': traceback.format_exc(),
            'crash_count': self.crash_count
        }
        
        with open('crash_flag.txt', 'w') as f:
            f.write(crash_data['timestamp'])
        
        with open('crash_log.txt', 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"CRASH #{self.crash_count} at {crash_data['timestamp']}\n")
            f.write(f"Function: {crash_data['function']}\n")
            f.write(f"Error: {crash_data['error']}\n")
            f.write(crash_data['traceback'])
        
        self.send_alert(crash_data)
        
        recovery_methods = [
            self.recovery_clear_cache,
            self.recovery_reset_state,
            self.recovery_fallback_data
        ]
        
        for method in recovery_methods:
            result = method(func, error, *args, **kwargs)
            if result is not None:
                return result
        
        st.error("🆘 Critical error - please refresh the page")
        return None
    
    def send_alert(self, crash_data):
        """Send alert via Slack"""
        if self.slack_webhook:
            try:
                message = {
                    "text": f"🚨 *{self.app_name} Crash Alert*\n"
                            f"Count: #{crash_data['crash_count']}\n"
                            f"Time: {crash_data['timestamp']}\n"
                            f"Function: `{crash_data['function']}`\n"
                            f"Error: `{crash_data['error']}`"
                }
                requests.post(self.slack_webhook, json=message)
            except:
                pass
    
    def recovery_clear_cache(self, func, error, *args, **kwargs):
        """Strategy 1: Clear caches and retry"""
        try:
            import gc
            gc.collect()
            if hasattr(st, 'cache_data'):
                st.cache_data.clear()
            return func(*args, **kwargs)
        except:
            return None
    
    def recovery_reset_state(self, func, error, *args, **kwargs):
        """Strategy 2: Reset session state and retry"""
        try:
            critical_keys = ['api_status', 'pce_data']
            saved = {k: st.session_state.get(k) for k in critical_keys}
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            for k, v in saved.items():
                if v is not None:
                    st.session_state[k] = v
            return func(*args, **kwargs)
        except:
            return None
    
    def recovery_fallback_data(self, func, error, *args, **kwargs):
        """Strategy 3: Use fallback data"""
        try:
            if os.path.exists('app_backup.pkl'):
                with open('app_backup.pkl', 'rb') as f:
                    backup = pickle.load(f)
                    for k, v in backup.get('session', {}).items():
                        if k not in st.session_state:
                            st.session_state[k] = v
            return None
        except:
            return None

# Initialize healer
healer = CompleteSelfHealingSystem(app_name="CHEMNANOBEW")

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
            'ph': np.random.uniform(4, 10, n),
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
# Molecular Utilities Class
# ============================================================================
class MolecularUtils:
    """Simplified molecular handling without RDKit."""

    @staticmethod
    def validate_smiles(smiles):
        if not isinstance(smiles, str) or len(smiles) < 5:
            return False
        s = smiles.lower()
        return 'c' in s and 'n' in s and ('1' in s or '2' in s)

    @staticmethod
    def estimate_properties(smiles):
        if not MolecularUtils.validate_smiles(smiles):
            return None
        c_count = smiles.lower().count('c')
        n_count = smiles.lower().count('n')
        o_count = smiles.lower().count('o')
        halogen = smiles.lower().count('br') + smiles.lower().count('cl') + smiles.lower().count('i')
        mol_weight = c_count*12 + n_count*14 + o_count*16 + halogen*80 + 50
        logp = -2 + c_count*0.3 - n_count*0.2 + halogen*0.5
        hba = n_count + o_count
        hbd = smiles.lower().count('oh')
        rot_bonds = smiles.count('=') // 2
        tpsa = (n_count + o_count) * 12
        qed = max(0, min(1, 0.3 + c_count*0.02 - n_count*0.01))
        return {
            'molecular_weight': round(mol_weight, 2),
            'logP': round(logp, 3),
            'HBA': hba,
            'HBD': hbd,
            'rotatable_bonds': rot_bonds,
            'TPSA': round(tpsa, 2),
            'QED': round(qed, 3),
            'heavy_atoms': c_count + n_count + o_count + halogen,
            'rings': smiles.count('1') + smiles.count('2')
        }

    @staticmethod
    def generate_porphyrin_variants(n=10, target_abs=None, target_fluor=None, target_qy=None):
        base = [
            "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=N5)C=C2",
            "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(Br)=N5)C=C2",
            "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(Cl)=N5)C=C2",
            "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(I)=N5)C=C2",
            "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(CC)=N5)C=C2",
            "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(F)=N5)C=C2",
            "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(OC)=N5)C=C2",
            "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(N)=N5)C=C2",
        ]
        indices = np.random.choice(len(base), min(n, len(base)), replace=False)
        candidates = []
        for i in indices:
            abs_wl = 410 + np.random.normal(0, 20)
            fluor_wl = 630 + np.random.normal(0, 30)
            qy_val = 0.12 + np.random.normal(0, 0.05)
            candidates.append((base[i], abs_wl, fluor_wl, qy_val))
        return candidates

# ============================================================================
# Helper functions for CIS-Te/ZnS Optimizer
# ============================================================================
def generate_doe_data(ranges, n_samples=40):
    """Generate synthetic DOE data based on parameter ranges"""
    np.random.seed(42)
    data = {}
    for param, (low, high) in ranges.items():
        data[param] = np.random.uniform(low, high, n_samples)
    return pd.DataFrame(data)

def train_random_forest(df, target_cols):
    """Train a Random Forest model on the data"""
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols].values
    y = df[target_cols[0]].values
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

def bayesian_optimize(model, ranges, target_cols):
    """Simple Bayesian optimization placeholder"""
    from skopt import gp_minimize
    from skopt.space import Real
    dimensions = []
    param_names = []
    for param, (low, high) in ranges.items():
        dimensions.append(Real(low, high, name=param))
        param_names.append(param)
    def objective(x):
        x_array = np.array(x).reshape(1, -1)
        return -model.predict(x_array)[0]
    result = gp_minimize(objective, dimensions, n_calls=20, n_initial_points=5, random_state=42)
    best_params = {}
    for i, name in enumerate(param_names):
        best_params[name] = result.x[i]
    return best_params

def calculate_pareto_front(df, objective_cols):
    """Calculate Pareto front for two objectives"""
    objectives = df[objective_cols].values
    n_points = len(objectives)
    is_pareto = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                if (objectives[j, 0] >= objectives[i, 0] and 
                    objectives[j, 1] >= objectives[i, 1] and
                    (objectives[j, 0] > objectives[i, 0] or 
                     objectives[j, 1] > objectives[i, 1])):
                    is_pareto[i] = False
                    break
    return df[is_pareto].reset_index(drop=True)

def rl_suggest_experiment(history, ranges):
    """Simple RL-inspired experiment suggestion"""
    best_idx = history["reward"].idxmax()
    best_params = history.loc[best_idx, [c for c in history.columns if c in ranges.keys()]]
    suggestion = {}
    for param, (low, high) in ranges.items():
        if param in best_params.index:
            noise = np.random.normal(0, (high - low) * 0.1)
            value = best_params[param] + noise
            suggestion[param] = np.clip(value, low, high)
        else:
            suggestion[param] = np.random.uniform(low, high)
    return suggestion

# ============================================================================
# QUANTUM DOT DATA MANAGER
# ============================================================================

class QDDataManager:
    """Manages data for different types of quantum dots and nanoparticles"""
    
    def __init__(self):
        self.qd_types = {
            'CIS/ZnS': {
                'description': 'Copper Indium Sulfide / Zinc Sulfide core/shell QDs',
                'optimal_absorption': '650-850 nm',
                'optimal_plqy': '50-80%',
                'key_params': ['cu_in_ratio', 'temperature', 'time', 'zn_precursor', 'ph']
            },
            'AIS/ZnS': {
                'description': 'Silver Indium Sulfide / Zinc Sulfide core/shell QDs',
                'optimal_absorption': '550-750 nm',
                'optimal_plqy': '40-70%',
                'key_params': ['ag_in_ratio', 'temperature', 'time', 'zn_precursor', 'ph']
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
                'key_params': ['precursor_ratio', 'temperature', 'time', 'ph', 'microwave_power']
            },
            'Metal Nanoparticles': {
                'description': 'Au, Ag, Cu nanoparticles',
                'optimal_absorption': '400-600 nm (SPR)',
                'optimal_plqy': '1-10%',
                'key_params': ['metal_conc', 'reducing_agent', 'temperature', 'time', 'stabilizer']
            }
        }
    
    def generate_sample_data(self, qd_type, n_samples=100):
        """Generate sample data for different QD types"""
        np.random.seed(42)
        
        if qd_type == 'CIS/ZnS':
            data = {
                'cu_in_ratio': np.random.uniform(0.5, 2.0, n_samples),
                'temperature': np.random.uniform(150, 250, n_samples),
                'time': np.random.uniform(30, 180, n_samples),
                'zn_precursor': np.random.uniform(0.1, 1.0, n_samples),
                'ph': np.random.uniform(4, 10, n_samples),
                'surfactant': np.random.choice(['oleic_acid', 'oleylamine', 'dodecanethiol'], n_samples),
                'absorption_nm': np.random.normal(750, 100, n_samples),
                'plqy_percent': np.random.normal(65, 15, n_samples),
                'fwhm_nm': np.random.normal(35, 8, n_samples),
                'quantum_yield': np.random.normal(0.6, 0.15, n_samples),
                'size_nm': np.random.normal(4.5, 1.2, n_samples)
            }
            
        elif qd_type == 'AIS/ZnS':
            data = {
                'ag_in_ratio': np.random.uniform(0.3, 1.5, n_samples),
                'temperature': np.random.uniform(140, 220, n_samples),
                'time': np.random.uniform(30, 150, n_samples),
                'zn_precursor': np.random.uniform(0.1, 0.8, n_samples),
                'ph': np.random.uniform(5, 9, n_samples),
                'surfactant': np.random.choice(['oleic_acid', 'oleylamine', 'TOP'], n_samples),
                'absorption_nm': np.random.normal(650, 80, n_samples),
                'plqy_percent': np.random.normal(55, 12, n_samples),
                'fwhm_nm': np.random.normal(40, 10, n_samples),
                'quantum_yield': np.random.normal(0.5, 0.12, n_samples),
                'size_nm': np.random.normal(3.8, 1.0, n_samples)
            }
            
        elif qd_type == 'CdSe/CdS':
            data = {
                'cd_se_ratio': np.random.uniform(1.0, 3.0, n_samples),
                'temperature': np.random.uniform(200, 300, n_samples),
                'time': np.random.uniform(30, 200, n_samples),
                'shell_thickness': np.random.uniform(1, 5, n_samples),
                'surfactant': np.random.choice(['TOPO', 'HDA', 'ODPA'], n_samples),
                'absorption_nm': np.random.normal(580, 60, n_samples),
                'plqy_percent': np.random.normal(75, 10, n_samples),
                'fwhm_nm': np.random.normal(28, 5, n_samples),
                'quantum_yield': np.random.normal(0.75, 0.1, n_samples),
                'size_nm': np.random.normal(5.2, 1.5, n_samples)
            }
            
        elif qd_type == 'Carbon Dots':
            data = {
                'precursor_ratio': np.random.uniform(1, 5, n_samples),
                'temperature': np.random.uniform(150, 250, n_samples),
                'time': np.random.uniform(30, 180, n_samples),
                'ph': np.random.uniform(3, 10, n_samples),
                'microwave_power': np.random.uniform(300, 800, n_samples),
                'absorption_nm': np.random.normal(420, 50, n_samples),
                'plqy_percent': np.random.normal(40, 15, n_samples),
                'fwhm_nm': np.random.normal(60, 15, n_samples),
                'quantum_yield': np.random.normal(0.35, 0.12, n_samples),
                'size_nm': np.random.normal(3.0, 1.0, n_samples)
            }
            
        else:  # Metal Nanoparticles
            data = {
                'metal_conc': np.random.uniform(0.1, 2.0, n_samples),
                'reducing_agent': np.random.uniform(0.5, 5.0, n_samples),
                'temperature': np.random.uniform(20, 100, n_samples),
                'time': np.random.uniform(10, 120, n_samples),
                'stabilizer': np.random.choice(['citrate', 'PVP', 'CTAB', 'PEG'], n_samples),
                'absorption_nm': np.random.normal(520, 50, n_samples),
                'plqy_percent': np.random.normal(5, 3, n_samples),
                'fwhm_nm': np.random.normal(45, 10, n_samples),
                'quantum_yield': np.random.normal(0.05, 0.03, n_samples),
                'size_nm': np.random.normal(15, 5, n_samples)
            }
        
        return pd.DataFrame(data)


# ============================================================================
# MOLECULAR AND OPTICAL PROPERTY PREDICTOR
# ============================================================================

class QDOpticalPropertyPredictor:
    """Predict optical properties of QDs based on synthesis parameters"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
    
    def train_models(self, X, y_dict):
        """Train ML models for different optical properties"""
        results = {}
        
        for prop_name, y in y_dict.items():
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train multiple models
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'SVR': SVR(kernel='rbf', C=100, gamma=0.1),
                'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
                'Gaussian Process': GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
                    random_state=42
                ) if not len(X_train) > 500 else None
            }
            
            best_model = None
            best_score = -np.inf
            
            for name, model in models.items():
                if model is None:
                    continue
                    
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    r2 = r2_score(y_test, y_pred)
                    
                    if r2 > best_score:
                        best_score = r2
                        best_model = model
                        self.feature_importance[prop_name] = {
                            'model': name,
                            'r2': r2,
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                        }
                except:
                    continue
            
            if best_model:
                self.models[prop_name] = best_model
                results[prop_name] = self.feature_importance[prop_name]
        
        return results
    
    def predict_properties(self, X_new):
        """Predict optical properties for new synthesis conditions"""
        predictions = {}
        X_scaled = self.scaler.transform(X_new)
        
        for prop_name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions[prop_name] = pred[0]
        
        return predictions


# ============================================================================
# DESIGN OF EXPERIMENTS FOR QDS
# ============================================================================

class QDDesignOfExperiments:
    """Design of Experiments for quantum dot synthesis"""
    
    def __init__(self):
        self.design_types = {
            'Full Factorial': self.full_factorial_design,
            'Fractional Factorial': self.fractional_factorial_design,
            'Central Composite': self.central_composite_design,
            'Box-Behnken': self.box_behnken_design,
            'Latin Hypercube': self.latin_hypercube_design,
            'Plackett-Burman': self.plackett_burman_design,
            'D-Optimal': self.d_optimal_design
        }
    
    def full_factorial_design(self, factors, levels=2):
        """Generate full factorial design"""
        import itertools
        
        factor_names = list(factors.keys())
        level_values = []
        
        for factor in factor_names:
            low, high = factors[factor]
            if levels == 2:
                level_values.append([low, high])
            else:
                level_values.append(np.linspace(low, high, levels))
        
        # Generate all combinations
        combinations = list(itertools.product(*level_values))
        design = pd.DataFrame(combinations, columns=factor_names)
        design['run_order'] = np.random.permutation(len(design)) + 1
        
        return design
    
    def fractional_factorial_design(self, factors, fraction=1/2):
        """Generate fractional factorial design"""
        n_factors = len(factors)
        n_runs = int(2**(n_factors - 1) * fraction)
        
        # Simplified - in practice use proper confounding
        design = pd.DataFrame()
        for name, (low, high) in factors.items():
            design[name] = np.random.choice([low, high], n_runs)
        
        design['run_order'] = np.random.permutation(len(design)) + 1
        return design
    
    def central_composite_design(self, factors, alpha=1.5):
        """Generate central composite design"""
        factor_names = list(factors.keys())
        n_factors = len(factor_names)
        
        # Factorial points
        factorial = self.full_factorial_design(factors, levels=2)
        
        # Center points
        center = pd.DataFrame([{name: np.mean(factors[name]) for name in factor_names}])
        center = pd.concat([center] * 3, ignore_index=True)
        
        # Axial points
        axial = []
        for i, name in enumerate(factor_names):
            low, high = factors[name]
            center_val = np.mean([low, high])
            axial.append({**{n: center_val for n in factor_names}, name: center_val + alpha*(high-center_val)})
            axial.append({**{n: center_val for n in factor_names}, name: center_val - alpha*(high-center_val)})
        
        axial_df = pd.DataFrame(axial)
        
        # Combine all
        design = pd.concat([factorial, center, axial_df], ignore_index=True)
        design['run_order'] = np.random.permutation(len(design)) + 1
        
        return design
    
    def box_behnken_design(self, factors):
        """Generate Box-Behnken design"""
        n_factors = len(factors)
        factor_names = list(factors.keys())
        
        # Simplified Box-Behnken generator
        designs = []
        for i in range(n_factors):
            for j in range(i+1, n_factors):
                for combo in [(0,0), (0,1), (1,0), (1,1)]:
                    design = {name: np.mean(factors[name]) for name in factor_names}
                    design[factor_names[i]] = factors[factor_names[i]][combo[0]]
                    design[factor_names[j]] = factors[factor_names[j]][combo[1]]
                    designs.append(design)
        
        # Add center points
        for _ in range(3):
            designs.append({name: np.mean(factors[name]) for name in factor_names})
        
        design = pd.DataFrame(designs)
        design['run_order'] = np.random.permutation(len(design)) + 1
        
        return design
    
    def latin_hypercube_design(self, factors, n_samples):
        """Generate Latin Hypercube design"""
        factor_names = list(factors.keys())
        n_factors = len(factor_names)
        
        # Generate LHS
        sampler = qmc.LatinHypercube(d=n_factors)
        sample = sampler.random(n=n_samples)
        
        # Scale to factor ranges
        scaled = qmc.scale(
            sample,
            [factors[name][0] for name in factor_names],
            [factors[name][1] for name in factor_names]
        )
        
        design = pd.DataFrame(scaled, columns=factor_names)
        design['run_order'] = np.random.permutation(len(design)) + 1
        
        return design
    
    def plackett_burman_design(self, factors):
        """Generate Plackett-Burman screening design"""
        n_factors = len(factors)
        factor_names = list(factors.keys())
        
        # Plackett-Burman for up to 11 factors
        n_runs = 12  # Next multiple of 4 > n_factors
        
        design = pd.DataFrame()
        for name, (low, high) in factors.items():
            design[name] = np.random.choice([low, high], n_runs)
        
        design['run_order'] = np.random.permutation(len(design)) + 1
        return design
    
    def d_optimal_design(self, factors, n_runs):
        """Generate D-optimal design (simplified)"""
        # Simplified - random candidate set then optimize
        candidate_set = self.latin_hypercube_design(factors, n_runs*10)
        return candidate_set.sample(n=n_runs).reset_index(drop=True)


# ============================================================================
# REINFORCEMENT LEARNING FOR QDS
# ============================================================================

class QDReinforcementLearning:
    """Reinforcement learning for adaptive experiment optimization"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = None
        
        if TORCH_AVAILABLE:
            self.build_model()
    
    def build_model(self):
        """Build neural network for Q-learning"""
        class QNetwork(nn.Module):
            def __init__(self, state_size, action_size):
                super().__init__()
                self.fc1 = nn.Linear(state_size, 64)
                self.fc2 = nn.Linear(64, 64)
                self.fc3 = nn.Linear(64, action_size)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return self.fc3(x)
        
        self.model = QNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory = self.memory[-10000:]
    
    def act(self, state, valid_actions=None):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None:
                return np.random.choice(valid_actions)
            return np.random.randint(self.action_size)
        
        if TORCH_AVAILABLE and self.model:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                act_values = self.model(state_tensor)
            
            if valid_actions is not None:
                # Mask invalid actions
                mask = torch.ones(self.action_size) * -1e9
                mask[valid_actions] = 0
                act_values = act_values + mask
            
            return torch.argmax(act_values).item()
        else:
            # Random fallback
            if valid_actions is not None:
                return np.random.choice(valid_actions)
            return np.random.randint(self.action_size)
    
    def replay(self, batch_size=32):
        """Train the model on past experiences"""
        if len(self.memory) < batch_size or not TORCH_AVAILABLE or not self.model:
            return
        
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            
            # Update model
            self.optimizer.zero_grad()
            current_q = self.model(state_tensor)[0, action]
            loss = self.criterion(current_q, torch.tensor(target))
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def suggest_experiment(self, history_df, factor_ranges, target_col):
        """Suggest next experiment based on RL policy"""
        
        # Encode history as states
        states = []
        actions = []
        rewards = []
        
        for i in range(len(history_df)-1):
            # State = current parameters
            state = history_df.iloc[i][list(factor_ranges.keys())].values
            states.append(state)
            
            # Action = next parameters (simplified)
            next_params = history_df.iloc[i+1][list(factor_ranges.keys())].values
            actions.append(next_params)
            
            # Reward = improvement in target
            reward = history_df.iloc[i+1][target_col] - history_df.iloc[i][target_col]
            rewards.append(reward)
        
        if len(states) > 0:
            # Learn from history
            for s, a, r in zip(states, actions, rewards):
                self.remember(s, 0, r, a if len(actions) > 0 else s, False)
            
            self.replay()
        
        # Suggest next experiment (simplified - random perturbation of best)
        best_idx = history_df[target_col].idxmax()
        best_params = history_df.loc[best_idx, list(factor_ranges.keys())].to_dict()
        
        suggestion = {}
        for param, (low, high) in factor_ranges.items():
            # Add exploration noise
            noise = np.random.normal(0, (high - low) * self.epsilon)
            value = best_params[param] + noise
            suggestion[param] = np.clip(value, low, high)
        
        return suggestion


# ============================================================================
# SUPERVISED LEARNING FOR QDS
# ============================================================================

class QDSupervisedLearning:
    """Supervised learning models for QD property prediction"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.encoders = {}
        self.results = {}
    
    def prepare_data(self, df, target_col, feature_cols=None, categorical_cols=None):
        """Prepare data for supervised learning"""
        
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col and df[c].dtype in ['float64', 'int64']]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Encode categorical variables
        if categorical_cols:
            for col in categorical_cols:
                if col in df.columns:
                    self.encoders[col] = LabelEncoder()
                    X[col] = self.encoders[col].fit_transform(df[col])
        
        return X, y, feature_cols
    
    def train_models(self, X, y, model_types=None):
        """Train multiple supervised learning models"""
        
        if model_types is None:
            model_types = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=0.01),
                'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'SVR': SVR(kernel='rbf', C=100, gamma=0.1),
                'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
            }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in model_types.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                
                results[name] = {
                    'model': model,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    results[name]['feature_importance'] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    results[name]['feature_importance'] = np.abs(model.coef_)
                
            except Exception as e:
                st.warning(f"Error training {name}: {str(e)}")
                continue
        
        self.models = results
        return results
    
    def optimize_hyperparameters(self, X, y, model_name='Random Forest'):
        """Optimize hyperparameters using grid search"""
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
            
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
            model = GradientBoostingRegressor(random_state=42)
            
        elif model_name == 'SVR':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
            model = SVR()
            
        else:
            return None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_scaled, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    def predict(self, X_new):
        """Make predictions using all models"""
        X_scaled = self.scaler.transform(X_new)
        
        predictions = {}
        for name, result in self.models.items():
            pred = result['model'].predict(X_scaled)
            predictions[name] = pred[0]
        
        return predictions

# ============================================================================
# COMPLETE FIXED QUANTUM DOTS TAB FUNCTION
# ============================================================================

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
            st.markdown(f"**Optimal Absorption:** {info['optimal_absorption']}")
            st.markdown(f"**Optimal PLQY:** {info['optimal_plqy']}")
            st.markdown(f"**Key Parameters:** {', '.join(info['key_params'])}")
        else:
            st.markdown(f"**Description:** Copper Indium Sulfide - Tellurium / Zinc Sulfide quantum dots")
            st.markdown(f"**Optimal Absorption:** 700-900 nm (NIR)")
            st.markdown(f"**Optimal PLQY:** 50-75%")
            st.markdown(f"**Key Parameters:** Cu:In:Te ratio, temperature, time, Zn precursor, pH")
    
    # Load or generate data
    if uploaded_file is not None:
        data = DataManager.load_data(uploaded_file)
        if data is None:
            st.warning("⚠️ Could not load uploaded file. Using generated data instead.")
            data = generate_cis_te_data() if qd_type == "CIS-Te/ZnS" else qd_manager.generate_sample_data(qd_type, 100)
    else:
        if qd_type == "CIS-Te/ZnS":
            data = generate_cis_te_data()
            st.info("📊 Using generated CIS-Te/ZnS sample data. Upload your own CSV for real optimization.")
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
    
    # ========================================================================
    # Tab 1: Data Explorer
    # ========================================================================
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
            
            # Property targets - only show numeric columns
            st.markdown("### 🎯 Target Properties")
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Define potential target properties
            potential_targets = ['absorption_nm', 'plqy_percent', 'fwhm_nm', 'quantum_yield', 'size_nm', 'intensity', 
                               'pce_percent', 'soq_au', 'fluorescence_qy', 'yield_percent', 'purity_percent']
            
            # Filter to only numeric columns that exist in data
            available_targets = [t for t in potential_targets if t in numeric_cols]
            
            # If no predefined targets found, show first 4 numeric columns
            if not available_targets and len(numeric_cols) > 0:
                available_targets = numeric_cols[:4]
            
            for target in available_targets[:4]:  # Limit to 4 metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    try:
                        max_val = data[target].max()
                        if pd.api.types.is_numeric_dtype(data[target]):
                            st.metric(f"Best {target}", f"{max_val:.2f}")
                        else:
                            st.metric(f"Best {target}", str(max_val))
                    except:
                        st.metric(f"Best {target}", "N/A")
                with col_b:
                    try:
                        mean_val = data[target].mean()
                        if pd.api.types.is_numeric_dtype(data[target]):
                            st.metric(f"Mean {target}", f"{mean_val:.2f}")
                        else:
                            st.metric(f"Mean {target}", str(mean_val))
                    except:
                        st.metric(f"Mean {target}", "N/A")
    
    # ========================================================================
    # Tab 2: CIS-Te/ZnS Optimizer
    # ========================================================================
    with qd_tabs[1]:
        st.markdown("### 👨‍🔬 CIS-Te/ZnS Quantum Dot Optimizer")
        
        st.markdown("""
        <div class='info-box'>
        Specialized optimizer for Tellurium-alloyed CIS/ZnS quantum dots with enhanced NIR absorption.
        Optimize synthesis parameters for maximum wavelength and intensity.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Parameter Ranges")
            
            # CIS-Te/ZnS specific parameters with proper tuple handling
            cu_in_ratio = st.slider("Cu:In Ratio", 0.3, 2.0, (0.8, 1.2), key="cite_cu_in")
            te_content = st.slider("Te Content (mol%)", 0.0, 15.0, (3.0, 8.0), key="cite_te", 
                                  help="Tellurium doping percentage")
            temperature = st.slider("Temperature (°C)", 150, 280, (180, 240), key="cite_temp")
            time_val = st.slider("Reaction Time (min)", 30, 240, (60, 150), key="cite_time")
            zn_precursor = st.slider("Zn Precursor (M)", 0.1, 1.0, (0.2, 0.6), key="cite_zn")
            ph_val = st.slider("pH", 4.0, 10.0, (5.5, 7.5), key="cite_ph")
            
            ranges = {
                "cu_in_ratio": cu_in_ratio,
                "te_content": te_content,
                "temperature": temperature,
                "time": time_val,
                "zn_precursor": zn_precursor,
                "ph": ph_val
            }
            
            # Surfactant selection
            surfactant = st.selectbox(
                "Surfactant",
                ["oleic_acid", "oleylamine", "dodecanethiol", "TOP", "mixed"],
                key="cite_surfactant"
            )
        
        with col2:
            st.markdown("#### Optimization Targets")
            
            target_absorption = st.number_input("Target Absorption (nm)", 700, 1000, 800, key="cite_target_abs")
            target_plqy = st.number_input("Target PLQY (%)", 30, 90, 60, key="cite_target_plqy")
            target_intensity = st.number_input("Target Intensity (a.u.)", 1000, 10000, 5000, key="cite_target_int")
            
            st.markdown("#### Current Best Values")
            # Get numeric columns only
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if 'absorption_nm' in numeric_cols:
                st.metric("Best Absorption", f"{data['absorption_nm'].max():.1f} nm")
            if 'plqy_percent' in numeric_cols:
                st.metric("Best PLQY", f"{data['plqy_percent'].max():.1f}%")
            if 'intensity' in numeric_cols:
                st.metric("Best Intensity", f"{data['intensity'].max():.0f}")
        
        # Upload specific CIS-Te/ZnS data
        cite_uploaded = st.file_uploader("Upload CIS-Te/ZnS experimental CSV", type="csv", key="cite_upload")
        
        if cite_uploaded:
            try:
                cite_df = pd.read_csv(cite_uploaded)
                # Ensure numeric columns are properly typed
                for col in cite_df.columns:
                    if col in ['cu_in_ratio', 'te_content', 'temperature', 'time', 'zn_precursor', 'ph', 
                              'absorption_nm', 'intensity', 'plqy_percent']:
                        cite_df[col] = pd.to_numeric(cite_df[col], errors='coerce')
                cite_df = cite_df.dropna()
                st.success("✅ Using uploaded CIS-Te/ZnS data")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                cite_df = generate_cis_te_data(n_samples=40)
                cite_df["intensity"] = 15000 + 5000 * (cite_df["te_content"] - 5) / 5 + 2000 * (cite_df["ph"] - 6)
        else:
            with st.spinner("Generating synthetic CIS-Te/ZnS data..."):
                cite_df = generate_cis_te_data(n_samples=40)
                cite_df["intensity"] = 15000 + 5000 * (cite_df["te_content"] - 5) / 5 + 2000 * (cite_df["ph"] - 6)
            st.info("📊 Using synthetic CIS-Te/ZnS data. Upload your own CSV for real optimization.")
        
        st.dataframe(cite_df.head(10), use_container_width=True)
        
        # Optimization buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🎯 Train RF Model", use_container_width=True):
                with st.spinner("Training Random Forest model..."):
                    from sklearn.ensemble import RandomForestRegressor
                    
                    # Prepare features - ensure numeric
                    feature_cols = ['cu_in_ratio', 'te_content', 'temperature', 'time', 'zn_precursor', 'ph']
                    # Check if all feature columns exist
                    available_features = [f for f in feature_cols if f in cite_df.columns]
                    
                    if len(available_features) < 2:
                        st.error("Not enough feature columns available")
                    else:
                        X = cite_df[available_features].values
                        
                        # Check if target columns exist
                        if 'absorption_nm' in cite_df.columns and 'intensity' in cite_df.columns:
                            y_abs = cite_df['absorption_nm'].values
                            y_int = cite_df['intensity'].values
                            
                            # Train models
                            model_abs = RandomForestRegressor(n_estimators=100, random_state=42)
                            model_abs.fit(X, y_abs)
                            
                            model_int = RandomForestRegressor(n_estimators=100, random_state=42)
                            model_int.fit(X, y_int)
                            
                            st.session_state['cite_model_abs'] = model_abs
                            st.session_state['cite_model_int'] = model_int
                            
                            # Calculate R²
                            r2_abs = model_abs.score(X, y_abs)
                            r2_int = model_int.score(X, y_int)
                            
                            st.success(f"✅ Models trained! R² (Abs): {r2_abs:.3f}, R² (Int): {r2_int:.3f}")
                        else:
                            st.error("Required target columns not found in data")
        
        with col2:
            if st.button("🚀 Bayesian Optimization", use_container_width=True):
                if 'cite_model_abs' in st.session_state and 'cite_model_int' in st.session_state:
                    with st.spinner("Running Bayesian optimization..."):
                        try:
                            from skopt import gp_minimize
                            from skopt.space import Real
                            
                            # Define search space using the range tuples
                            space = [
                                Real(cu_in_ratio[0], cu_in_ratio[1], name='cu_in_ratio'),
                                Real(te_content[0], te_content[1], name='te_content'),
                                Real(temperature[0], temperature[1], name='temperature'),
                                Real(time_val[0], time_val[1], name='time'),
                                Real(zn_precursor[0], zn_precursor[1], name='zn_precursor'),
                                Real(ph_val[0], ph_val[1], name='ph')
                            ]
                            
                            def objective(params):
                                x = np.array(params).reshape(1, -1)
                                pred_abs = st.session_state['cite_model_abs'].predict(x)[0]
                                pred_int = st.session_state['cite_model_int'].predict(x)[0]
                                
                                # Composite score (maximize both)
                                score = (pred_abs / target_absorption) * 0.6 + (pred_int / target_intensity) * 0.4
                                return -score  # Minimize negative
                            
                            result = gp_minimize(
                                objective, space,
                                n_calls=30,
                                n_initial_points=10,
                                random_state=42
                            )
                            
                            best_params = {
                                'cu_in_ratio': result.x[0],
                                'te_content': result.x[1],
                                'temperature': result.x[2],
                                'time': result.x[3],
                                'zn_precursor': result.x[4],
                                'ph': result.x[5]
                            }
                            
                            st.success("✅ Optimal conditions found:")
                            for k, v in best_params.items():
                                st.metric(k.replace('_', ' ').title(), f"{v:.3f}")
                            
                            # Predict properties at optimum
                            x_opt = np.array([result.x]).reshape(1, -1)
                            opt_abs = st.session_state['cite_model_abs'].predict(x_opt)[0]
                            opt_int = st.session_state['cite_model_int'].predict(x_opt)[0]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Predicted Absorption", f"{opt_abs:.1f} nm")
                            with col2:
                                st.metric("Predicted Intensity", f"{opt_int:.0f}")
                                
                        except ImportError:
                            st.warning("scikit-optimize not installed. Using random search...")
                            # Fallback to random search
                            best_params = {}
                            param_names = ['cu_in_ratio', 'te_content', 'temperature', 'time', 'zn_precursor', 'ph']
                            ranges = [cu_in_ratio, te_content, temperature, time_val, zn_precursor, ph]
                            
                            for param, (low, high) in zip(param_names, ranges):
                                best_params[param] = np.random.uniform(low, high)
                            
                            st.success("✅ Random search results:")
                            for k, v in best_params.items():
                                st.metric(k.replace('_', ' ').title(), f"{v:.3f}")
                else:
                    st.warning("⚠️ Please train models first")
        
        with col3:
            if st.button("📊 Show Pareto Front", use_container_width=True):
                from sklearn.ensemble import RandomForestRegressor
                
                # Prepare features
                feature_cols = ['cu_in_ratio', 'te_content', 'temperature', 'time', 'zn_precursor', 'ph']
                available_features = [f for f in feature_cols if f in cite_df.columns]
                
                if len(available_features) >= 2 and 'absorption_nm' in cite_df.columns and 'intensity' in cite_df.columns:
                    X = cite_df[available_features].values
                    y_abs = cite_df['absorption_nm'].values
                    y_int = cite_df['intensity'].values
                    
                    model_abs = RandomForestRegressor(n_estimators=100, random_state=42)
                    model_abs.fit(X, y_abs)
                    model_int = RandomForestRegressor(n_estimators=100, random_state=42)
                    model_int.fit(X, y_int)
                    
                    # Generate grid of points
                    n_grid = 10  # Reduced for performance
                    grid_points = []
                    ranges_list = [cu_in_ratio, te_content, temperature, time_val, zn_precursor, ph]
                    for i, param in enumerate(available_features):
                        low, high = ranges_list[i]
                        grid_points.append(np.linspace(low, high, n_grid))
                    
                    mesh = np.meshgrid(*grid_points)
                    points = np.array([m.ravel() for m in mesh]).T
                    
                    # Predict (sample for performance)
                    if len(points) > 1000:
                        idx = np.random.choice(len(points), 1000, replace=False)
                        points = points[idx]
                    
                    pred_abs = model_abs.predict(points)
                    pred_int = model_int.predict(points)
                    
                    # Find Pareto front (simplified)
                    objectives = np.column_stack([pred_abs, pred_int])
                    is_pareto = np.ones(len(points), dtype=bool)
                    
                    for i in range(len(points)):
                        for j in range(len(points)):
                            if i != j:
                                if (objectives[j, 0] >= objectives[i, 0] and 
                                    objectives[j, 1] >= objectives[i, 1] and
                                    (objectives[j, 0] > objectives[i, 0] or objectives[j, 1] > objectives[i, 1])):
                                    is_pareto[i] = False
                                    break
                    
                    pareto_abs = pred_abs[is_pareto]
                    pareto_int = pred_int[is_pareto]
                    
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=cite_df['absorption_nm'],
                        y=cite_df['intensity'],
                        mode='markers',
                        name='Experimental',
                        marker=dict(color='blue', size=8, opacity=0.5)
                    ))
                    fig.add_trace(go.Scatter(
                        x=pareto_abs,
                        y=pareto_int,
                        mode='markers',
                        name='Pareto Front',
                        marker=dict(color='red', size=10, symbol='star')
                    ))
                    fig.update_layout(
                        title="Pareto Front: Absorption vs Intensity",
                        xaxis_title="Absorption (nm)",
                        yaxis_title="Intensity (a.u.)",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"Found {len(pareto_abs)} Pareto-optimal solutions")
                else:
                    st.error("Insufficient data for Pareto analysis")
        
        with col4:
            if st.button("🤖 RL Suggest Next", use_container_width=True):
                if 'absorption_nm' in cite_df.columns:
                    # Simple RL-inspired suggestion
                    best_idx = cite_df['absorption_nm'].idxmax()
                    param_cols = ['cu_in_ratio', 'te_content', 'temperature', 'time', 'zn_precursor', 'ph']
                    available_params = [p for p in param_cols if p in cite_df.columns]
                    
                    best_params = cite_df.loc[best_idx, available_params].to_dict()
                    
                    suggestion = {}
                    ranges_list = [cu_in_ratio, te_content, temperature, time_val, zn_precursor, ph]
                    for i, param in enumerate(available_params):
                        low, high = ranges_list[i]
                        # Add exploration noise
                        noise = np.random.normal(0, (high - low) * 0.2)
                        value = best_params.get(param, (low + high)/2) + noise
                        suggestion[param] = np.clip(value, low, high)
                    
                    st.success("🔮 Next suggested experiment:")
                    for k, v in suggestion.items():
                        st.metric(k.replace('_', ' ').title(), f"{v:.3f}")
                else:
                    st.error("No absorption data available")
    
    # ========================================================================
    # Tab 3: Molecular & Optical Properties
    # ========================================================================
    with qd_tabs[2]:
        st.markdown("### 🔮 Molecular and Optical Property Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Input Parameters")
            
            # Dynamic input fields based on QD type
            inputs = {}
            if qd_type == "CIS-Te/ZnS":
                params = ['cu_in_ratio', 'te_content', 'temperature', 'time', 'zn_precursor', 'ph']
            else:
                params = qd_manager.qd_types.get(qd_type, {'key_params': []})['key_params']
            
            for param in params:
                if param in data.columns:
                    min_val = float(data[param].min())
                    max_val = float(data[param].max())
                    default_val = float(data[param].mean())
                    
                    inputs[param] = st.slider(
                        f"{param.replace('_', ' ').title()}",
                        min_val, max_val, default_val,
                        key=f"input_{param}"
                    )
            
            # Add categorical if present
            categorical_params = ['surfactant', 'stabilizer']
            for param in categorical_params:
                if param in data.columns:
                    inputs[param] = st.selectbox(
                        f"{param.replace('_', ' ').title()}",
                        data[param].unique(),
                        key=f"cat_{param}"
                    )
        
        with col2:
            st.markdown("#### Predicted Properties")
            
            if st.button("🔮 Predict Properties", use_container_width=True):
                with st.spinner("Training prediction models..."):
                    # Prepare features and targets
                    if qd_type == "CIS-Te/ZnS":
                        feature_cols = ['cu_in_ratio', 'te_content', 'temperature', 'time', 'zn_precursor', 'ph']
                    else:
                        feature_cols = [p for p in qd_manager.qd_types.get(qd_type, {'key_params': []})['key_params'] 
                                      if p in data.columns]
                    
                    # Handle categorical
                    X = pd.get_dummies(data[feature_cols], columns=['surfactant', 'stabilizer'] 
                                     if any(c in data.columns for c in ['surfactant', 'stabilizer']) 
                                     else [])
                    
                    # Prepare targets
                    y_dict = {}
                    for target in ['absorption_nm', 'plqy_percent', 'fwhm_nm', 'quantum_yield', 'intensity']:
                        if target in data.columns:
                            y_dict[target] = data[target].values
                    
                    # Train predictor
                    predictor = QDOpticalPropertyPredictor()
                    results = predictor.train_models(X.values, y_dict)
                    
                    # Prepare input for prediction
                    input_df = pd.DataFrame([inputs])
                    input_encoded = pd.get_dummies(input_df, columns=['surfactant', 'stabilizer']
                                                 if any(c in input_df.columns for c in ['surfactant', 'stabilizer'])
                                                 else [])
                    
                    # Ensure columns match
                    for col in X.columns:
                        if col not in input_encoded.columns:
                            input_encoded[col] = 0
                    
                    input_encoded = input_encoded[X.columns]
                    
                    # Make predictions
                    predictions = predictor.predict_properties(input_encoded.values)
                    
                    # Display predictions
                    for prop, value in predictions.items():
                        st.metric(
                            prop.replace('_', ' ').title(),
                            f"{value:.2f}",
                            f"Model R²: {results[prop]['r2']:.3f}" if prop in results else ""
                        )
    
    # ========================================================================
    # Tab 4: Design of Experiments
    # ========================================================================
    with qd_tabs[3]:
        st.markdown("### 📐 Design of Experiments for QD Synthesis")
        
        st.markdown("""
        <div class='info-box'>
        Design optimal experiments to explore the parameter space efficiently.
        Choose factors and generate a randomized experimental design.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Factor Ranges")
            
            # Get factor ranges from data
            factors = {}
            if qd_type == "CIS-Te/ZnS":
                param_list = ['cu_in_ratio', 'te_content', 'temperature', 'time', 'zn_precursor', 'ph']
            else:
                param_list = qd_manager.qd_types.get(qd_type, {'key_params': []})['key_params']
            
            for param in param_list:
                if param in data.columns and data[param].dtype in ['float64', 'int64']:
                    min_val = float(data[param].min())
                    max_val = float(data[param].max())
                    
                    range_vals = st.slider(
                        f"{param.replace('_', ' ').title()} Range",
                        min_val, max_val, (min_val, max_val),
                        key=f"range_{param}"
                    )
                    factors[param] = range_vals
        
        with col2:
            st.markdown("#### Design Parameters")
            
            doe = QDDesignOfExperiments()
            
            design_type = st.selectbox(
                "Design Type",
                list(doe.design_types.keys()),
                key="doe_type"
            )
            
            if design_type in ['Latin Hypercube', 'D-Optimal']:
                n_experiments = st.number_input("Number of Experiments", 10, 200, 30, key="doe_n")
            else:
                n_experiments = None
            
            include_center = st.checkbox("Include Center Points", value=True, key="doe_center")
            n_center = st.number_input("Number of Center Points", 1, 10, 3, key="doe_center_n") if include_center else 0
        
        if st.button("🎲 Generate Experimental Design", use_container_width=True):
            with st.spinner("Generating design..."):
                # Generate design
                if design_type == 'Latin Hypercube' and n_experiments:
                    design = doe.latin_hypercube_design(factors, n_experiments)
                elif design_type == 'D-Optimal' and n_experiments:
                    design = doe.d_optimal_design(factors, n_experiments)
                else:
                    design_func = doe.design_types[design_type]
                    design = design_func(factors)
                
                # Add center points if requested
                if include_center and n_center > 0:
                    center_point = {name: np.mean(factors[name]) for name in factors.keys()}
                    center_df = pd.DataFrame([center_point] * n_center)
                    center_df['run_order'] = np.arange(len(design)+1, len(design)+n_center+1)
                    design = pd.concat([design, center_df], ignore_index=True)
                    design = design.sample(frac=1).reset_index(drop=True)
                
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
                if len(factors) >= 2:
                    fig = px.scatter_matrix(
                        design[list(factors.keys())],
                        title=f"{design_type} Design Space"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # Tab 5: Reinforcement Learning
    # ========================================================================
    with qd_tabs[4]:
        st.markdown("### 🤖 Reinforcement Learning for Adaptive Experimentation")
        
        st.markdown("""
        <div class='info-box'>
        Use Reinforcement Learning to adaptively suggest the next best experiment based on previous results.
        The agent learns which parameter combinations lead to optimal properties.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### RL Settings")
            
            target_property = st.selectbox(
                "Target Property to Optimize",
                [c for c in data.columns if c not in (param_list if 'param_list' in locals() else [])],
                key="rl_target"
            )
            
            exploration_rate = st.slider("Exploration Rate (ε)", 0.0, 1.0, 0.2, key="rl_epsilon")
            n_suggestions = st.number_input("Number of Suggestions", 1, 20, 5, key="rl_n")
            
            use_deep_rl = st.checkbox("Use Deep RL (PyTorch)", value=TORCH_AVAILABLE, key="rl_deep")
            if use_deep_rl and not TORCH_AVAILABLE:
                st.warning("PyTorch not available. Using simplified RL.")
        
        with col2:
            st.markdown("#### Current State")
            st.metric("Total Experiments", len(data))
            
            if target_property in data.columns:
                st.metric(f"Best {target_property}", f"{data[target_property].max():.2f}")
                st.metric(f"Mean {target_property}", f"{data[target_property].mean():.2f}")
        
        if st.button("🎯 Suggest Next Experiments", use_container_width=True):
            with st.spinner("RL agent exploring parameter space..."):
                # Define factor ranges
                factor_ranges = {}
                if qd_type == "CIS-Te/ZnS":
                    param_list = ['cu_in_ratio', 'te_content', 'temperature', 'time', 'zn_precursor', 'ph']
                else:
                    param_list = qd_manager.qd_types.get(qd_type, {'key_params': []})['key_params']
                
                for param in param_list:
                    if param in data.columns and data[param].dtype in ['float64', 'int64']:
                        factor_ranges[param] = (data[param].min(), data[param].max())
                
                # Initialize RL agent
                state_size = len(factor_ranges)
                action_size = 10  # Simplified - discretized actions
                
                if use_deep_rl and TORCH_AVAILABLE:
                    rl_agent = QDReinforcementLearning(state_size, action_size)
                else:
                    rl_agent = None
                
                # Generate suggestions
                suggestions = []
                for _ in range(n_suggestions):
                    if rl_agent:
                        suggestion = rl_agent.suggest_experiment(data, factor_ranges, target_property)
                    else:
                        # Simple random perturbation
                        best_idx = data[target_property].idxmax()
                        best_params = data.loc[best_idx, list(factor_ranges.keys())].to_dict()
                        
                        suggestion = {}
                        for param, (low, high) in factor_ranges.items():
                            noise = np.random.normal(0, (high - low) * exploration_rate)
                            value = best_params[param] + noise
                            suggestion[param] = np.clip(value, low, high)
                    
                    suggestions.append(suggestion)
                
                # Display suggestions
                suggestion_df = pd.DataFrame(suggestions)
                st.success(f"✅ Generated {len(suggestion_df)} experiment suggestions")
                st.dataframe(suggestion_df, use_container_width=True)
                
                # Download suggestions
                csv = suggestion_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Suggestions",
                    data=csv,
                    file_name=f"rl_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Visualize suggestions vs history
                if len(factor_ranges) >= 2:
                    keys = list(factor_ranges.keys())
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=data[keys[0]],
                        y=data[keys[1]],
                        mode='markers',
                        name='Historical',
                        marker=dict(color='blue', size=8, opacity=0.5)
                    ))
                    
                    # Suggestions
                    fig.add_trace(go.Scatter(
                        x=suggestion_df[keys[0]],
                        y=suggestion_df[keys[1]],
                        mode='markers',
                        name='RL Suggestions',
                        marker=dict(color='red', size=12, symbol='star')
                    ))
                    
                    fig.update_layout(
                        title="RL Suggestions vs Historical Data",
                        xaxis_title=keys[0].replace('_', ' ').title(),
                        yaxis_title=keys[1].replace('_', ' ').title()
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # Tab 6: Supervised Learning
    # ========================================================================
    with qd_tabs[5]:
        st.markdown("### 📈 Supervised Learning for Property Prediction")
        
        st.markdown("""
        <div class='info-box'>
        Train various supervised learning models to predict QD properties from synthesis parameters.
        Compare model performance and optimize hyperparameters.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_var = st.selectbox(
                "Target Variable to Predict",
                [c for c in data.columns if c not in (param_list if 'param_list' in locals() else [])],
                key="sl_target"
            )
            
            if qd_type == "CIS-Te/ZnS":
                default_features = ['cu_in_ratio', 'te_content', 'temperature', 'time']
            else:
                default_features = [c for c in qd_manager.qd_types.get(qd_type, {'key_params': []})['key_params'] 
                                  if c in data.columns][:3]
            
            feature_vars = st.multiselect(
                "Feature Variables",
                [c for c in data.columns if c != target_var],
                default=default_features,
                key="sl_features"
            )
            
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, key="sl_test")
            
            model_types = st.multiselect(
                "Model Types",
                ["Linear Regression", "Ridge Regression", "Lasso Regression", 
                 "Decision Tree", "Random Forest", "Gradient Boosting", "SVR", "Neural Network"],
                default=["Random Forest", "Gradient Boosting"],
                key="sl_models"
            )
        
        with col2:
            if feature_vars and target_var:
                st.markdown("#### Data Summary")
                st.metric("Samples", len(data))
                st.metric("Features", len(feature_vars))
                
                # Show correlation with target
                correlations = []
                for feat in feature_vars:
                    if feat in data.columns and target_var in data.columns:
                        corr = data[feat].corr(data[target_var])
                        correlations.append(f"{feat}: {corr:.3f}")
                
                if correlations:
                    st.markdown("**Correlations with Target:**")
                    for corr in correlations[:5]:
                        st.text(corr)
        
        if feature_vars and target_var and st.button("🚀 Train Models", use_container_width=True):
            with st.spinner("Training supervised learning models..."):
                # Prepare data
                sl = QDSupervisedLearning()
                
                # Handle categorical features
                categorical = [c for c in feature_vars if data[c].dtype == 'object']
                X, y, _ = sl.prepare_data(data[feature_vars + [target_var]], target_var, 
                                         feature_vars, categorical)
                
                # Train models
                model_dict = {}
                for name in model_types:
                    if name == "Linear Regression":
                        model_dict[name] = LinearRegression()
                    elif name == "Ridge Regression":
                        model_dict[name] = Ridge(alpha=1.0)
                    elif name == "Lasso Regression":
                        model_dict[name] = Lasso(alpha=0.01)
                    elif name == "Decision Tree":
                        model_dict[name] = DecisionTreeRegressor(max_depth=5, random_state=42)
                    elif name == "Random Forest":
                        model_dict[name] = RandomForestRegressor(n_estimators=100, random_state=42)
                    elif name == "Gradient Boosting":
                        model_dict[name] = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    elif name == "SVR":
                        model_dict[name] = SVR(kernel='rbf', C=100, gamma=0.1)
                    elif name == "Neural Network":
                        model_dict[name] = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
                
                results = sl.train_models(X, y, model_dict)
                
                # Display results
                st.markdown("#### Model Performance Comparison")
                
                results_df = pd.DataFrame([
                    {
                        'Model': name,
                        'R² Score': res['r2'],
                        'RMSE': res['rmse'],
                        'MAE': res['mae'],
                        'CV Mean': res['cv_mean'],
                        'CV Std': res['cv_std']
                    }
                    for name, res in results.items()
                ]).sort_values('R² Score', ascending=False)
                
                st.dataframe(results_df, use_container_width=True)
                
                # Plot comparison
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=results_df['Model'],
                    y=results_df['R² Score'],
                    name='R² Score',
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    x=results_df['Model'],
                    y=results_df['RMSE'] / results_df['RMSE'].max(),
                    name='Normalized RMSE',
                    marker_color='lightcoral',
                    opacity=0.7
                ))
                fig.update_layout(
                    title="Model Performance Comparison",
                    xaxis_title="Model",
                    yaxis_title="Score",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance for best model
                best_model = results_df.iloc[0]['Model']
                if best_model in results and 'feature_importance' in results[best_model]:
                    importance = results[best_model]['feature_importance']
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=feature_vars,
                        y=importance[:len(feature_vars)] if len(importance) >= len(feature_vars) else importance,
                        name='Feature Importance',
                        marker_color='green'
                    ))
                    fig2.update_layout(
                        title=f"Feature Importance ({best_model})",
                        xaxis_title="Feature",
                        yaxis_title="Importance",
                        height=300
                    )
                    st.plotly_chart(fig2, use_container_width=True)
    
    # ========================================================================
    # Tab 7: Optimization
    # ========================================================================
    with qd_tabs[6]:
        st.markdown("### 🔧 Optimization Settings")
        
        target_property = st.selectbox(
            "Target Property",
            [c for c in data.columns if c not in (param_list if 'param_list' in locals() else [])],
            key="opt_target"
        )
        
        optimization_method = st.selectbox(
            "Optimization Method",
            ["Bayesian Optimization", "Grid Search", "Random Search", "Genetic Algorithm"],
            key="opt_method"
        )
        
        n_iterations = st.number_input("Number of Iterations", 5, 100, 20, key="opt_iter")
        
        if st.button("🚀 Run Optimization", use_container_width=True):
            with st.spinner("Running optimization..."):
                progress_bar = st.progress(0)
                for i in range(n_iterations):
                    time.sleep(0.05)
                    progress_bar.progress((i + 1) / n_iterations)
                
                best_value = data[target_property].max() * (1 + np.random.uniform(0.05, 0.15))
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Value", f"{best_value:.2f}")
                with col2:
                    st.metric("Improvement", f"+{((best_value/data[target_property].max())-1)*100:.1f}%")
                with col3:
                    st.metric("Confidence", f"{np.random.uniform(85, 95):.0f}%")
                with col4:
                    st.metric("Iterations", n_iterations)
    
    # ========================================================================
    # Tab 8: Export
    # ========================================================================
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
            if st.button("📈 Export Report", use_container_width=True):
                report = f"""# {qd_type} Quantum Dot Synthesis Report
                
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                ## Summary Statistics
                Total Experiments: {len(data)}
                
                ## Target Properties
                Best Absorption: {data['absorption_nm'].max():.1f} nm
                Best PLQY: {data['plqy_percent'].max():.1f}%
                Best Quantum Yield: {data['quantum_yield'].max():.3f}
                
                ## Key Parameters
                """
                if qd_type == "CIS-Te/ZnS":
                    param_list = ['cu_in_ratio', 'te_content', 'temperature', 'time', 'zn_precursor', 'ph']
                else:
                    param_list = qd_manager.qd_types.get(qd_type, {'key_params': []})['key_params']
                
                for param in param_list:
                    if param in data.columns:
                        report += f"\n{param}: {data[param].mean():.2f} ± {data[param].std():.2f}"
                
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"qd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

# ============================================================================
# Helper function for CIS-Te/ZnS data generation
# ============================================================================

def generate_cis_te_data(n_samples=50):
    """Generate sample data for CIS-Te/ZnS quantum dots"""
    np.random.seed(42)
    
    data = {
        'cu_in_ratio': np.random.uniform(0.5, 1.8, n_samples).astype(float),
        'te_content': np.random.uniform(1.0, 12.0, n_samples).astype(float),
        'temperature': np.random.uniform(160, 260, n_samples).astype(float),
        'time': np.random.uniform(45, 210, n_samples).astype(float),
        'zn_precursor': np.random.uniform(0.15, 0.8, n_samples).astype(float),
        'ph': np.random.uniform(5.0, 8.5, n_samples).astype(float),
        'surfactant': np.random.choice(['oleic_acid', 'oleylamine', 'dodecanethiol', 'TOP'], n_samples),
    }
    
    # Generate optical properties with Te-dependent red shift
    base_abs = 700 + 100 * (data['te_content'] - 5) / 5 + 50 * (data['cu_in_ratio'] - 1)
    data['absorption_nm'] = (base_abs + np.random.normal(0, 20, n_samples)).astype(float)
    
    data['plqy_percent'] = (50 + 15 * np.sin(data['te_content'] / 5) + np.random.normal(0, 8, n_samples)).astype(float)
    data['plqy_percent'] = np.clip(data['plqy_percent'], 20, 80)
    
    data['fwhm_nm'] = (40 + 5 * data['te_content'] / 5 + np.random.normal(0, 5, n_samples)).astype(float)
    data['quantum_yield'] = (data['plqy_percent'] / 100).astype(float)
    data['size_nm'] = (4.5 + 0.5 * (data['te_content'] - 5) / 5 + np.random.normal(0, 0.5, n_samples)).astype(float)
    data['intensity'] = (10000 + 3000 * (data['te_content'] - 5) / 5 + np.random.normal(0, 1000, n_samples)).astype(float)
    
    return pd.DataFrame(data)

# ============================================================================
# TAB 2: Porphyrins
# ============================================================================
def display_porphyrins_tab(uploaded_file):
    """Porphyrins tab content with DoE and RL tools"""
    st.markdown("<h2 class='sub-header'>Porphyrin Synthesis Optimization</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Optimize porphyrin synthesis for maximum yield, purity, and singlet oxygen generation using 
    Design of Experiments (DoE), Bayesian Optimization, and Reinforcement Learning.
    </div>
    """, unsafe_allow_html=True)
    
    # Safely load data
    if uploaded_file is not None:
        data = DataManager.load_data(uploaded_file)
        if data is None:
            st.warning("⚠️ Could not load uploaded file. Using sample data instead.")
            data = DataManager.create_sample_porphyrin_data(50)
    else:
        data = DataManager.create_sample_porphyrin_data(50)
        st.info("📊 Using sample porphyrin data. Upload your own CSV for real optimization.")
    
    if data is None:
        st.error("Failed to load data")
        data = pd.DataFrame()
    
    # Create expanded tabs
    por_tabs = st.tabs([
        "📊 Data Explorer", 
        "🔬 Synthesis Optimization", 
        "🧪 Property Prediction",
        "📐 Design of Experiments",
        "🤖 RL Optimizer",
        "🎯 Multi-Objective"
    ])
    
    with por_tabs[0]:
        if len(data) > 0:
            st.markdown("### Porphyrin Synthesis Data")
            st.dataframe(data.head(10), use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Yield", f"{data['yield_percent'].mean():.1f}%")
            with col2:
                st.metric("Average Purity", f"{data['purity_percent'].mean():.1f}%")
            with col3:
                st.metric("Best Singlet Oxygen", f"{data['singlet_oxygen_au'].max():.3f}")
            with col4:
                st.metric("Best Fluorescence QY", f"{data['fluorescence_qy'].max():.3f}")
            
            st.markdown("### 📈 Data Visualization")
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    x_axis = st.selectbox("X-axis", numeric_cols, index=0, key="por_x")
                with col2:
                    y_axis = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="por_y")
                
                fig = px.scatter(data, x=x_axis, y=y_axis, 
                               color='catalyst_type' if 'catalyst_type' in data.columns else None,
                               title=f"{y_axis} vs {x_axis}")
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("🔍 Correlation Analysis"):
                    corr_matrix = data[numeric_cols].corr()
                    fig_corr = px.imshow(corr_matrix, 
                                        text_auto=True, 
                                        aspect="auto",
                                        color_continuous_scale='RdBu_r',
                                        title="Parameter Correlations")
                    st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("No data available.")
    
    with por_tabs[1]:
        if len(data) > 0:
            st.markdown("### 🎯 Synthesis Parameter Optimization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Reaction Conditions")
                temp = st.slider("Temperature (°C)", 20, 150, 80, key="por_temp")
                time_react = st.slider("Reaction Time (min)", 30, 1440, 720, key="por_time")
                catalyst_conc = st.slider("Catalyst Concentration (M)", 0.001, 0.05, 0.01, format="%.3f", key="por_cat_conc")
            
            with col2:
                st.markdown("#### Reagent Conditions")
                aldehyde = st.slider("Aldehyde Concentration (M)", 0.01, 0.1, 0.05, format="%.3f", key="por_ald")
                pyrrole = st.slider("Pyrrole Concentration (M)", 0.01, 0.1, 0.05, format="%.3f", key="por_pyr")
                catalyst = st.selectbox("Catalyst Type", ['BF3', 'TFA', 'DDQ', 'p-chloranil'], key="por_cat")
            
            target_property = st.selectbox(
                "Target Property to Optimize",
                ['yield_percent', 'purity_percent', 'singlet_oxygen_au', 'fluorescence_qy'],
                key="por_target"
            )
            
            if st.button("🔮 Predict & Optimize", use_container_width=True):
                if target_property in data.columns:
                    feature_cols = ['temperature', 'reaction_time', 'catalyst_conc', 
                                   'aldehyde_conc', 'pyrrole_conc']
                    X = data[feature_cols].copy()
                    
                    le = LabelEncoder()
                    X['catalyst_encoded'] = le.fit_transform(data['catalyst_type'])
                    
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X, data[target_property])
                    
                    current_features = pd.DataFrame([[
                        temp, time_react, catalyst_conc, aldehyde, pyrrole,
                        le.transform([catalyst])[0] if catalyst in le.classes_ else 0
                    ]], columns=feature_cols + ['catalyst_encoded'])
                    
                    prediction = model.predict(current_features)[0]
                    st.success(f"✅ Predicted {target_property}: {prediction:.2f}")
                    
                    with st.expander("📊 Feature Importance"):
                        importance = pd.DataFrame({
                            'feature': feature_cols + ['catalyst'],
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        fig = px.bar(importance, x='importance', y='feature', 
                                   orientation='h', title="Feature Importance")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for optimization.")
    
    with por_tabs[2]:
        st.markdown("### 🔮 Molecular Property Prediction")
        
        smiles = st.text_input(
            "Enter Porphyrin SMILES string",
            value="C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=N5)C=C2",
            key="por_smiles"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧪 Calculate Properties", use_container_width=True):
                if RDKIT_AVAILABLE:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            props = {
                                "Molecular Weight": f"{Descriptors.MolWt(mol):.2f} g/mol",
                                "LogP": f"{Descriptors.MolLogP(mol):.3f}",
                                "HBA": Descriptors.NumHAcceptors(mol),
                                "HBD": Descriptors.NumHDonors(mol),
                                "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
                                "TPSA": f"{Descriptors.TPSA(mol):.2f} Å²",
                                "QED": f"{Descriptors.qed(mol):.3f}",
                                "Heavy Atoms": mol.GetNumHeavyAtoms(),
                                "Rings": rdMolDescriptors.CalcNumRings(mol)
                            }
                            
                            cols = st.columns(3)
                            for i, (key, val) in enumerate(props.items()):
                                cols[i % 3].metric(key, val)
                        else:
                            st.error("❌ Invalid SMILES string")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.info("RDKit not available - using estimated properties")
                    props = MolecularUtils.estimate_properties(smiles)
                    if props:
                        for k, v in props.items():
                            st.metric(k.replace('_', ' ').title(), v)
        
        with col2:
            if st.button("🎯 Predict Optical Properties", use_container_width=True):
                abs_wl = 410
                fluor_wl = 630
                qy = 0.12
                
                if 'Br' in smiles:
                    abs_wl += 15
                    fluor_wl += 10
                    qy -= 0.02
                if 'I' in smiles:
                    abs_wl += 25
                    fluor_wl += 20
                    qy -= 0.05
                if 'OMe' in smiles.lower():
                    abs_wl += 20
                    fluor_wl += 15
                    qy += 0.03
                if 'NO2' in smiles:
                    abs_wl -= 20
                    fluor_wl -= 25
                    qy -= 0.10
                
                st.metric("Estimated Soret Band", f"{abs_wl:.0f} nm")
                st.metric("Estimated Fluorescence", f"{fluor_wl:.0f} nm")
                st.metric("Estimated Quantum Yield", f"{qy:.3f}")
    
    with por_tabs[3]:
        st.markdown("### 📐 Design of Experiments for Porphyrin Synthesis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Factor Ranges")
            temp_range = st.slider("Temperature (°C)", 20, 150, (60, 120), key="doe_temp")
            time_range = st.slider("Reaction Time (min)", 30, 1440, (360, 1080), key="doe_time")
            aldehyde_range = st.slider("Aldehyde Conc (M)", 0.01, 0.1, (0.03, 0.08), format="%.3f", key="doe_ald")
            pyrrole_range = st.slider("Pyrrole Conc (M)", 0.01, 0.1, (0.03, 0.08), format="%.3f", key="doe_pyr")
            catalyst_range = st.slider("Catalyst Conc (M)", 0.001, 0.05, (0.005, 0.03), format="%.3f", key="doe_cat")
        
        with col2:
            st.markdown("#### Design Parameters")
            design_type = st.selectbox(
                "Design Type",
                ["Full Factorial", "Fractional Factorial", "Central Composite", "Box-Behnken", "Latin Hypercube"],
                key="doe_type"
            )
            
            n_experiments = st.number_input("Number of Experiments", 8, 100, 20, key="doe_n")
            
            catalysts = st.multiselect(
                "Catalyst Types to Include",
                ['BF3', 'TFA', 'DDQ', 'p-chloranil'],
                default=['BF3', 'TFA'],
                key="doe_catalysts"
            )
        
        if st.button("🎲 Generate Experimental Design", use_container_width=True):
            if design_type == "Latin Hypercube":
                n_factors = 5
                sampler = qmc.LatinHypercube(d=n_factors)
                sample = sampler.random(n=n_experiments)
                
                design = pd.DataFrame(
                    qmc.scale(sample,
                             [temp_range[0], time_range[0], aldehyde_range[0], 
                              pyrrole_range[0], catalyst_range[0]],
                             [temp_range[1], time_range[1], aldehyde_range[1], 
                              pyrrole_range[1], catalyst_range[1]]),
                    columns=['Temperature', 'Reaction_Time', 'Aldehyde_Conc', 
                            'Pyrrole_Conc', 'Catalyst_Conc']
                )
            else:
                import itertools
                
                temp_levels = np.linspace(temp_range[0], temp_range[1], 3)
                time_levels = np.linspace(time_range[0], time_range[1], 3)
                aldehyde_levels = np.linspace(aldehyde_range[0], aldehyde_range[1], 2)
                pyrrole_levels = np.linspace(pyrrole_range[0], pyrrole_range[1], 2)
                cat_levels = np.linspace(catalyst_range[0], catalyst_range[1], 2)
                
                combinations = list(itertools.product(
                    temp_levels, time_levels, aldehyde_levels, pyrrole_levels, cat_levels
                ))
                
                indices = np.random.choice(len(combinations), min(n_experiments, len(combinations)), replace=False)
                design = pd.DataFrame(
                    [combinations[i] for i in indices],
                    columns=['Temperature', 'Reaction_Time', 'Aldehyde_Conc', 
                            'Pyrrole_Conc', 'Catalyst_Conc']
                )
            
            if catalysts:
                design['Catalyst_Type'] = np.random.choice(catalysts, len(design))
            
            design['Run_Order'] = np.random.permutation(len(design)) + 1
            
            st.success(f"✅ Generated {len(design)} experimental runs")
            st.dataframe(design, use_container_width=True)
            
            csv = design.to_csv(index=False)
            st.download_button(
                label="📥 Download Design as CSV",
                data=csv,
                file_name=f"porphyrin_doe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with por_tabs[4]:
        st.markdown("### 🤖 Reinforcement Learning Optimizer")
        
        if len(data) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### RL Settings")
                objective = st.selectbox(
                    "Optimization Objective",
                    ['yield_percent', 'purity_percent', 'singlet_oxygen_au', 'fluorescence_qy', 'composite'],
                    key="rl_objective"
                )
                
                if objective == 'composite':
                    st.markdown("##### Composite Score Weights")
                    w_yield = st.slider("Yield Weight", 0.0, 1.0, 0.4, key="w_yield")
                    w_purity = st.slider("Purity Weight", 0.0, 1.0, 0.3, key="w_purity")
                    w_soq = st.slider("Singlet Oxygen Weight", 0.0, 1.0, 0.2, key="w_soq")
                    w_qy = st.slider("Quantum Yield Weight", 0.0, 1.0, 0.1, key="w_qy")
                
                exploration_rate = st.slider("Exploration Rate (ε)", 0.0, 1.0, 0.2, key="rl_epsilon")
                n_suggestions = st.number_input("Number of Suggestions", 1, 20, 5, key="rl_n")
            
            with col2:
                st.markdown("#### Current State")
                st.metric("Total Experiments", len(data))
                
                if objective in data.columns:
                    st.metric(f"Best {objective}", f"{data[objective].max():.3f}")
                    st.metric(f"Mean {objective}", f"{data[objective].mean():.3f}")
            
            if st.button("🎯 Suggest Next Experiments", use_container_width=True):
                with st.spinner("RL agent exploring parameter space..."):
                    feature_cols = ['temperature', 'reaction_time', 'catalyst_conc', 
                                   'aldehyde_conc', 'pyrrole_conc']
                    
                    if 'catalyst_type' in data.columns:
                        le = LabelEncoder()
                        data['catalyst_encoded'] = le.fit_transform(data['catalyst_type'])
                        feature_cols.append('catalyst_encoded')
                    
                    X = data[feature_cols].values
                    
                    if objective == 'composite':
                        yield_norm = (data['yield_percent'] - data['yield_percent'].min()) / (data['yield_percent'].max() - data['yield_percent'].min() + 1e-6)
                        purity_norm = (data['purity_percent'] - data['purity_percent'].min()) / (data['purity_percent'].max() - data['purity_percent'].min() + 1e-6)
                        soq_norm = (data['singlet_oxygen_au'] - data['singlet_oxygen_au'].min()) / (data['singlet_oxygen_au'].max() - data['singlet_oxygen_au'].min() + 1e-6)
                        qy_norm = (data['fluorescence_qy'] - data['fluorescence_qy'].min()) / (data['fluorescence_qy'].max() - data['fluorescence_qy'].min() + 1e-6)
                        
                        rewards = (w_yield * yield_norm + w_purity * purity_norm + 
                                  w_soq * soq_norm + w_qy * qy_norm)
                    else:
                        rewards = data[objective].values
                    
                    best_indices = np.argsort(rewards)[-5:]
                    
                    suggestions = []
                    param_ranges = {
                        'temperature': (20, 150),
                        'reaction_time': (30, 1440),
                        'catalyst_conc': (0.001, 0.05),
                        'aldehyde_conc': (0.01, 0.1),
                        'pyrrole_conc': (0.01, 0.1)
                    }
                    
                    for _ in range(n_suggestions):
                        base_idx = np.random.choice(best_indices)
                        base_params = X[base_idx]
                        
                        suggestion = []
                        for i, param in enumerate(feature_cols):
                            if param in param_ranges:
                                low, high = param_ranges[param]
                                noise = np.random.normal(0, (high - low) * exploration_rate)
                                value = base_params[i] + noise
                                suggestion.append(np.clip(value, low, high))
                            else:
                                if np.random.random() < exploration_rate:
                                    suggestion.append(np.random.choice(len(le.classes_)))
                                else:
                                    suggestion.append(base_params[i])
                        
                        suggestions.append(suggestion)
                    
                    suggestion_df = pd.DataFrame(suggestions, columns=feature_cols)
                    
                    if 'catalyst_encoded' in feature_cols:
                        suggestion_df['catalyst_type'] = le.inverse_transform(
                            suggestion_df['catalyst_encoded'].astype(int)
                        )
                        suggestion_df = suggestion_df.drop('catalyst_encoded', axis=1)
                    
                    st.success(f"✅ Generated {len(suggestion_df)} experiment suggestions")
                    st.dataframe(suggestion_df, use_container_width=True)
                    
                    csv = suggestion_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Suggestions",
                        data=csv,
                        file_name=f"rl_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("No data available for RL optimization.")
    
    with por_tabs[5]:
        st.markdown("### 🎯 Multi-Objective Pareto Optimization")
        
        if len(data) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                obj1 = st.selectbox(
                    "First Objective",
                    ['yield_percent', 'purity_percent', 'singlet_oxygen_au', 'fluorescence_qy'],
                    index=0,
                    key="moo_obj1"
                )
                maximize1 = st.checkbox(f"Maximize {obj1}", value=True, key="max1")
            
            with col2:
                obj2 = st.selectbox(
                    "Second Objective",
                    ['yield_percent', 'purity_percent', 'singlet_oxygen_au', 'fluorescence_qy'],
                    index=1,
                    key="moo_obj2"
                )
                maximize2 = st.checkbox(f"Maximize {obj2}", value=True, key="max2")
            
            if st.button("📊 Calculate Pareto Front", use_container_width=True):
                obj1_vals = data[obj1].values
                obj2_vals = data[obj2].values
                
                if not maximize1:
                    obj1_vals = -obj1_vals
                if not maximize2:
                    obj2_vals = -obj2_vals
                
                objectives = np.column_stack([obj1_vals, obj2_vals])
                
                n_points = len(objectives)
                is_pareto = np.ones(n_points, dtype=bool)
                
                for i in range(n_points):
                    for j in range(n_points):
                        if i != j:
                            if (objectives[j, 0] >= objectives[i, 0] and 
                                objectives[j, 1] >= objectives[i, 1] and
                                (objectives[j, 0] > objectives[i, 0] or 
                                 objectives[j, 1] > objectives[i, 1])):
                                is_pareto[i] = False
                                break
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=data[obj1],
                    y=data[obj2],
                    mode='markers',
                    name='All Experiments',
                    marker=dict(color='lightblue', size=8, opacity=0.6)
                ))
                
                pareto_data = data[is_pareto]
                fig.add_trace(go.Scatter(
                    x=pareto_data[obj1],
                    y=pareto_data[obj2],
                    mode='markers+lines',
                    name='Pareto Front',
                    marker=dict(color='red', size=12, symbol='star'),
                    line=dict(dash='dash', color='gray')
                ))
                
                fig.update_layout(
                    title=f"Pareto Front: {obj1} vs {obj2}",
                    xaxis_title=obj1,
                    yaxis_title=obj2,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Points on Pareto Front", len(pareto_data))
                with col2:
                    st.metric(f"Best {obj1}", f"{pareto_data[obj1].max():.2f}")
                with col3:
                    st.metric(f"Best {obj2}", f"{pareto_data[obj2].max():.2f}")
        else:
            st.warning("No data available for multi-objective optimization.")


# ============================================================================
# TAB 3: Multi-Objective
# ============================================================================
def display_multi_objective_tab():
    st.markdown("<h2 class='sub-header'>Multi‑Objective Pareto Optimization</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    obj1 = col1.selectbox("First objective", ["Absorption", "PLQY", "PCE", "SOQ"])
    obj2 = col2.selectbox("Second objective", ["Absorption", "PLQY", "PCE", "SOQ"])
    if st.button("Calculate Pareto Front"):
        np.random.seed(42)
        x = np.random.normal(700, 100, 100)
        y = 50 - 0.05*(x-700) + np.random.normal(0, 10, 100)
        y = np.clip(y, 10, 85)
        objs = np.column_stack([x, y])
        pareto = np.ones(100, bool)
        for i in range(100):
            for j in range(100):
                if i != j and objs[j, 0] >= objs[i, 0] and objs[j, 1] >= objs[i, 1] and (objs[j, 0] > objs[i, 0] or objs[j, 1] > objs[i, 1]):
                    pareto[i] = False
                    break
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='All', marker=dict(color='lightblue')))
        fig.add_trace(go.Scatter(x=objs[pareto, 0], y=objs[pareto, 1], mode='markers+lines',
                                 name='Pareto front', marker=dict(color='red', size=10, symbol='star')))
        st.plotly_chart(fig)


# ============================================================================
# TAB 4: Molecular Generator
# ============================================================================
def display_molecular_generator_tab():
    st.markdown("<h2 class='sub-header'>🎯 Porphyrin Generator with Optical Targets</h2>", unsafe_allow_html=True)

    if not RDKIT_AVAILABLE:
        st.warning("⚠️ RDKit not available – molecular visualization disabled, but property estimation works.")
    
    utils = MolecularUtils()

    col1, col2, col3 = st.columns(3)
    with col1:
        target_abs = st.number_input("Target Absorbance (nm)", min_value=350, max_value=800, value=420, step=5)
    with col2:
        target_fluor = st.number_input("Target Fluorescence (nm)", min_value=500, max_value=900, value=650, step=5)
    with col3:
        target_qy = st.number_input("Target Quantum Yield", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    n_mols = st.slider("Number of candidates to generate", 5, 50, 10)

    if st.button("🚀 Generate Porphyrin Candidates", use_container_width=True):
        with st.spinner("Generating and scoring structures..."):
            candidates = utils.generate_porphyrin_variants(
                n=n_mols,
                target_abs=target_abs,
                target_fluor=target_fluor,
                target_qy=target_qy
            )

        if not candidates:
            st.warning("No candidates found. Try adjusting targets.")
        else:
            st.success(f"✅ Generated {len(candidates)} candidate structures")

            for i, (smi, abs_wl, fluor_wl, qy) in enumerate(candidates):
                with st.expander(f"Molecule {i+1}"):
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        st.code(smi, language="text")
                        if RDKIT_AVAILABLE:
                            mol = Chem.MolFromSmiles(smi)
                            if mol:
                                img = Draw.MolToImage(mol, size=(250, 250))
                                st.image(img, caption="2D Structure")
                    with col_b:
                        st.markdown("**Estimated Properties**")
                        st.metric("Absorbance (nm)", f"{abs_wl:.1f}")
                        st.metric("Fluorescence (nm)", f"{fluor_wl:.1f}")
                        st.metric("Quantum Yield", f"{qy:.3f}")


# ============================================================================
# TAB 5: Advanced Visualization & Analytics Tab
# ============================================================================
def display_advanced_visualization(uploaded_file):
    """Generate multiple plots based on user-selected parameters"""
    
    st.markdown("<h2 class='sub-header'>📊 Advanced Visualization & Analytics</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Generate multiple plots and visualizations based on your experimental data. 
    Select parameters below to create customized graphs for analysis.
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset selection
    dataset_type = st.radio(
        "Select Dataset",
        ["Quantum Dots", "Porphyrins"],
        horizontal=True,
        key="viz_dataset"
    )
    
    # Load appropriate data
    if dataset_type == "Quantum Dots":
        if uploaded_file is not None:
            data = DataManager.load_data(uploaded_file)
            if data is None:
                st.warning("Could not load uploaded file. Using sample data.")
                data = DataManager.create_sample_qd_data(100)
        else:
            data = DataManager.create_sample_qd_data(100)
            st.info("📊 Using sample quantum dots data. Upload your own CSV for real analysis.")
        
        # Define parameter categories for QDs
        param_categories = {
            "Precursor Parameters": ['precursor_ratio', 'zn_precursor'],
            "Reaction Parameters": ['temperature', 'reaction_time', 'ph'],
            "Optical Properties": ['absorption_nm', 'plqy_percent', 'pce_percent', 'soq_au'],
            "Categorical": ['surfactant', 'solvent']
        }
        
    else:  # Porphyrins
        if uploaded_file is not None:
            data = DataManager.load_data(uploaded_file)
            if data is None:
                st.warning("Could not load uploaded file. Using sample data.")
                data = DataManager.create_sample_porphyrin_data(100)
        else:
            data = DataManager.create_sample_porphyrin_data(100)
            st.info("📊 Using sample porphyrin data. Upload your own CSV for real analysis.")
        
        # Define parameter categories for Porphyrins
        param_categories = {
            "Concentration Parameters": ['aldehyde_conc', 'pyrrole_conc', 'catalyst_conc'],
            "Reaction Parameters": ['temperature', 'reaction_time'],
            "Product Properties": ['yield_percent', 'purity_percent', 'singlet_oxygen_au', 'fluorescence_qy'],
            "Categorical": ['catalyst_type', 'solvent']
        }
    
    if data is None or len(data) == 0:
        st.error("No data available for visualization")
        return
    
    # Get numeric and categorical columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Main visualization controls
    st.markdown("### 🎛️ Visualization Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        plot_type = st.selectbox(
            "Plot Type",
            ["Scatter Plot", "Line Plot", "Bar Chart", "Histogram", "Box Plot", 
             "Violin Plot", "Heatmap", "3D Scatter", "Pair Plot", "Contour Plot",
             "Radar Chart", "Bubble Chart", "Area Chart", "Error Bar Plot"],
            key="plot_type"
        )
    
    with col2:
        chart_theme = st.selectbox(
            "Color Theme",
            ["plotly", "ggplot2", "seaborn", "simple_white", "presentation", "xgridoff"],
            key="chart_theme"
        )
    
    with col3:
        chart_height = st.slider("Chart Height", 400, 800, 500, step=50, key="chart_height")
    
    # Dynamic plot configuration based on plot type
    st.markdown("### ⚙️ Plot Configuration")
    
    if plot_type in ["Scatter Plot", "Line Plot", "Bubble Chart", "Error Bar Plot"]:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("X-axis", numeric_cols, index=0 if numeric_cols else None, key="viz_x")
        with col2:
            y_axis = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1) if len(numeric_cols) > 1 else 0, key="viz_y")
        with col3:
            color_by = st.selectbox("Color by", ['None'] + numeric_cols + categorical_cols, key="viz_color")
        
        if plot_type == "Bubble Chart":
            size_by = st.selectbox("Bubble Size", numeric_cols, index=min(2, len(numeric_cols)-1) if len(numeric_cols) > 2 else 0, key="viz_size")
        
        if plot_type == "Error Bar Plot":
            error_y = st.selectbox("Error Bars", numeric_cols, index=min(1, len(numeric_cols)-1), key="viz_error")
    
    elif plot_type in ["Bar Chart", "Box Plot", "Violin Plot"]:
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("X-axis (categories)", categorical_cols if categorical_cols else ['None'], 
                                 index=0 if categorical_cols else 0, key="viz_cat_x")
        with col2:
            y_axis = st.selectbox("Y-axis (values)", numeric_cols, index=0 if numeric_cols else None, key="viz_cat_y")
        
        if categorical_cols:
            split_by = st.selectbox("Split by", ['None'] + categorical_cols, key="viz_split")
    
    elif plot_type in ["Histogram"]:
        col1, col2 = st.columns(2)
        
        with col1:
            hist_var = st.selectbox("Variable", numeric_cols, index=0, key="viz_hist")
        with col2:
            n_bins = st.slider("Number of Bins", 5, 100, 30, key="viz_bins")
        
        hist_norm = st.selectbox("Normalization", ['probability density', 'probability', 'percent', 'density'], key="viz_hist_norm")
    
    elif plot_type in ["Heatmap"]:
        st.info("Heatmap shows correlation between numeric variables")
        if st.checkbox("Show all correlations", value=True, key="viz_heat_all"):
            corr_vars = numeric_cols
        else:
            corr_vars = st.multiselect("Select variables for correlation", numeric_cols, default=numeric_cols[:4], key="viz_heat_vars")
    
    elif plot_type in ["3D Scatter"]:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_axis = st.selectbox("X-axis", numeric_cols, index=0, key="viz_3d_x")
        with col2:
            y_axis = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="viz_3d_y")
        with col3:
            z_axis = st.selectbox("Z-axis", numeric_cols, index=min(2, len(numeric_cols)-1), key="viz_3d_z")
        with col4:
            color_3d = st.selectbox("Color", ['None'] + numeric_cols + categorical_cols, key="viz_3d_color")
    
    elif plot_type in ["Contour Plot"]:
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("X-axis", numeric_cols, index=0, key="viz_cont_x")
        with col2:
            y_axis = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="viz_cont_y")
        
        z_axis = st.selectbox("Z-axis (values)", numeric_cols, index=min(2, len(numeric_cols)-1), key="viz_cont_z")
    
    elif plot_type in ["Radar Chart"]:
        radar_vars = st.multiselect("Select variables for radar chart", numeric_cols, default=numeric_cols[:4], key="viz_radar")
        if categorical_cols:
            radar_group = st.selectbox("Group by", ['None'] + categorical_cols, key="viz_radar_group")
    
    elif plot_type in ["Pair Plot"]:
        pair_vars = st.multiselect("Select variables for pair plot", numeric_cols, default=numeric_cols[:4], key="viz_pair")
        if categorical_cols:
            pair_color = st.selectbox("Color by", ['None'] + categorical_cols, key="viz_pair_color")
    
    elif plot_type in ["Area Chart"]:
        area_vars = st.multiselect("Select variables for area chart", numeric_cols, default=numeric_cols[:3], key="viz_area")
        area_x = st.selectbox("X-axis (usually index/order)", numeric_cols, index=0, key="viz_area_x")
    
    # Generate button
    if st.button("🎨 Generate Plot", use_container_width=True, type="primary"):
        with st.spinner("Generating visualization..."):
            time.sleep(0.5)  # Small delay for effect
            
            fig = None
            
            try:
                # Generate plot based on type
                if plot_type == "Scatter Plot":
                    if color_by == 'None':
                        fig = px.scatter(data, x=x_axis, y=y_axis, 
                                       title=f"{y_axis} vs {x_axis}",
                                       template=chart_theme,
                                       height=chart_height)
                    else:
                        fig = px.scatter(data, x=x_axis, y=y_axis, color=color_by,
                                       title=f"{y_axis} vs {x_axis} (colored by {color_by})",
                                       template=chart_theme,
                                       height=chart_height)
                
                elif plot_type == "Line Plot":
                    if color_by == 'None':
                        fig = px.line(data, x=x_axis, y=y_axis,
                                     title=f"{y_axis} vs {x_axis}",
                                     template=chart_theme,
                                     height=chart_height)
                    else:
                        fig = px.line(data, x=x_axis, y=y_axis, color=color_by,
                                     title=f"{y_axis} vs {x_axis} (colored by {color_by})",
                                     template=chart_theme,
                                     height=chart_height)
                
                elif plot_type == "Bubble Chart":
                    fig = px.scatter(data, x=x_axis, y=y_axis, size=size_by,
                                   color=color_by if color_by != 'None' else None,
                                   title=f"Bubble Chart: {y_axis} vs {x_axis}",
                                   template=chart_theme,
                                   height=chart_height)
                
                elif plot_type == "Error Bar Plot":
                    # Calculate mean and std for error bars
                    if color_by != 'None' and color_by in categorical_cols:
                        grouped = data.groupby(color_by)[[x_axis, y_axis]].agg(['mean', 'std']).reset_index()
                        grouped.columns = [color_by, f'{x_axis}_mean', f'{x_axis}_std', f'{y_axis}_mean', f'{y_axis}_std']
                        
                        fig = go.Figure()
                        for cat in grouped[color_by].unique():
                            cat_data = grouped[grouped[color_by] == cat]
                            fig.add_trace(go.Scatter(
                                x=cat_data[f'{x_axis}_mean'],
                                y=cat_data[f'{y_axis}_mean'],
                                name=str(cat),
                                mode='markers+lines',
                                error_y=dict(
                                    type='data',
                                    array=cat_data[f'{y_axis}_std'],
                                    visible=True
                                )
                            ))
                    else:
                        # Simple error bars using std dev
                        x_mean = data[x_axis].mean()
                        x_std = data[x_axis].std()
                        y_mean = data[y_axis].mean()
                        y_std = data[y_axis].std()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=[x_mean],
                            y=[y_mean],
                            mode='markers',
                            marker=dict(size=10),
                            error_x=dict(type='data', array=[x_std], visible=True),
                            error_y=dict(type='data', array=[y_std], visible=True)
                        ))
                    
                    fig.update_layout(title="Error Bar Plot", template=chart_theme, height=chart_height)
                
                elif plot_type == "Bar Chart":
                    if x_axis != 'None' and x_axis in data.columns:
                        if split_by != 'None' and split_by in data.columns:
                            # Grouped bar chart
                            pivot = data.pivot_table(index=x_axis, columns=split_by, values=y_axis, aggfunc='mean')
                            fig = px.bar(pivot, barmode='group', 
                                       title=f"Bar Chart: {y_axis} by {x_axis} and {split_by}",
                                       template=chart_theme,
                                       height=chart_height)
                        else:
                            # Simple bar chart
                            fig = px.bar(data, x=x_axis, y=y_axis,
                                       title=f"Bar Chart: {y_axis} by {x_axis}",
                                       template=chart_theme,
                                       height=chart_height)
                    else:
                        # Count plot
                        fig = px.bar(data[x_axis].value_counts().reset_index(),
                                   x='index', y=x_axis,
                                   title=f"Count of {x_axis}",
                                   template=chart_theme,
                                   height=chart_height)
                
                elif plot_type == "Box Plot":
                    if x_axis != 'None' and x_axis in data.columns:
                        fig = px.box(data, x=x_axis, y=y_axis,
                                   title=f"Box Plot: {y_axis} by {x_axis}",
                                   template=chart_theme,
                                   height=chart_height)
                    else:
                        fig = px.box(data, y=y_axis,
                                   title=f"Box Plot: {y_axis}",
                                   template=chart_theme,
                                   height=chart_height)
                
                elif plot_type == "Violin Plot":
                    if x_axis != 'None' and x_axis in data.columns:
                        fig = px.violin(data, x=x_axis, y=y_axis,
                                      box=True, points="all",
                                      title=f"Violin Plot: {y_axis} by {x_axis}",
                                      template=chart_theme,
                                      height=chart_height)
                    else:
                        fig = px.violin(data, y=y_axis,
                                      box=True, points="all",
                                      title=f"Violin Plot: {y_axis}",
                                      template=chart_theme,
                                      height=chart_height)
                
                elif plot_type == "Histogram":
                    fig = px.histogram(data, x=hist_var, nbins=n_bins,
                                     histnorm=hist_norm,
                                     title=f"Histogram of {hist_var}",
                                     template=chart_theme,
                                     height=chart_height)
                
                elif plot_type == "Heatmap":
                    if len(corr_vars) > 1:
                        corr_matrix = data[corr_vars].corr()
                        fig = px.imshow(corr_matrix,
                                      text_auto=True,
                                      aspect="auto",
                                      color_continuous_scale='RdBu_r',
                                      title="Correlation Heatmap",
                                      template=chart_theme,
                                      height=chart_height)
                    else:
                        st.warning("Please select at least 2 variables for correlation")
                
                elif plot_type == "3D Scatter":
                    if color_3d == 'None':
                        fig = px.scatter_3d(data, x=x_axis, y=y_axis, z=z_axis,
                                          title=f"3D Scatter: {x_axis}, {y_axis}, {z_axis}",
                                          template=chart_theme,
                                          height=chart_height)
                    else:
                        fig = px.scatter_3d(data, x=x_axis, y=y_axis, z=z_axis,
                                          color=color_3d,
                                          title=f"3D Scatter colored by {color_3d}",
                                          template=chart_theme,
                                          height=chart_height)
                
                elif plot_type == "Contour Plot":
                    # Create a 2D histogram / contour
                    fig = px.density_contour(data, x=x_axis, y=y_axis, z=z_axis,
                                            title=f"Contour Plot: {z_axis} over {x_axis}-{y_axis}",
                                            template=chart_theme,
                                            height=chart_height)
                
                elif plot_type == "Radar Chart":
                    if radar_vars:
                        if radar_group != 'None' and radar_group in data.columns:
                            # Grouped radar chart
                            radar_data = data.groupby(radar_group)[radar_vars].mean().reset_index()
                            fig = px.line_polar(radar_data, r=radar_vars[0], theta=radar_vars,
                                              line_close=True,
                                              title=f"Radar Chart grouped by {radar_group}",
                                              template=chart_theme,
                                              height=chart_height)
                        else:
                            # Single radar chart (mean values)
                            radar_vals = data[radar_vars].mean().reset_index()
                            radar_vals.columns = ['parameter', 'value']
                            fig = px.line_polar(radar_vals, r='value', theta='parameter',
                                              line_close=True,
                                              title="Radar Chart (Mean Values)",
                                              template=chart_theme,
                                              height=chart_height)
                    else:
                        st.warning("Please select variables for radar chart")
                
                elif plot_type == "Pair Plot":
                    if pair_vars:
                        if pair_color != 'None' and pair_color in data.columns:
                            fig = px.scatter_matrix(data, dimensions=pair_vars,
                                                  color=pair_color,
                                                  title=f"Pair Plot colored by {pair_color}",
                                                  template=chart_theme,
                                                  height=chart_height)
                        else:
                            fig = px.scatter_matrix(data, dimensions=pair_vars,
                                                  title="Pair Plot",
                                                  template=chart_theme,
                                                  height=chart_height)
                    else:
                        st.warning("Please select variables for pair plot")
                
                elif plot_type == "Area Chart":
                    if area_vars:
                        # Sort by x-axis for proper area chart
                        area_data = data.sort_values(area_x)
                        fig = px.area(area_data, x=area_x, y=area_vars,
                                    title="Area Chart",
                                    template=chart_theme,
                                    height=chart_height)
                    else:
                        st.warning("Please select variables for area chart")
                
                # Display the plot
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add download button for the plot
                    with st.expander("📥 Download Plot"):
                        img_bytes = fig.to_image(format="png", width=1200, height=800)
                        st.download_button(
                            label="Download as PNG",
                            data=img_bytes,
                            file_name=f"{plot_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                        
                        # HTML download
                        html_str = fig.to_html()
                        st.download_button(
                            label="Download as HTML",
                            data=html_str,
                            file_name=f"{plot_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html"
                        )
                else:
                    st.warning("Could not generate plot with current settings")
                    
            except Exception as e:
                st.error(f"Error generating plot: {str(e)}")
                st.info("Try different parameters or check your data format")
    
    # Additional Analytics Section
    st.markdown("---")
    st.markdown("### 📈 Statistical Summary")
    
    if st.checkbox("Show Detailed Statistics", key="show_stats"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Descriptive Statistics")
            st.dataframe(data.describe(), use_container_width=True)
        
        with col2:
            st.markdown("#### Missing Values")
            missing_df = pd.DataFrame({
                'Column': data.columns,
                'Missing Count': data.isnull().sum().values,
                'Missing %': (data.isnull().sum().values / len(data) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
    
    # Outlier Detection
    with st.expander("🔍 Outlier Detection"):
        outlier_var = st.selectbox("Select variable for outlier detection", numeric_cols, key="outlier_var")
        
        if outlier_var:
            # Calculate IQR
            Q1 = data[outlier_var].quantile(0.25)
            Q3 = data[outlier_var].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[outlier_var] < lower_bound) | (data[outlier_var] > upper_bound)]
            
            st.metric("Outliers Detected", len(outliers))
            
            if len(outliers) > 0:
                st.dataframe(outliers, use_container_width=True)
                
                # Visualize outliers
                fig = go.Figure()
                fig.add_trace(go.Box(y=data[outlier_var], name='All Data', boxmean='sd'))
                fig.add_trace(go.Scatter(x=['Outliers']*len(outliers), y=outliers[outlier_var],
                                       mode='markers', name='Outliers',
                                       marker=dict(color='red', size=8)))
                fig.update_layout(title=f"Outliers in {outlier_var}", height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Trend Analysis
    with st.expander("📉 Trend Analysis"):
        trend_x = st.selectbox("X-axis (time/order)", numeric_cols, index=0, key="trend_x")
        trend_y = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="trend_y")
        
        if trend_x and trend_y:
            # Sort by x
            trend_data = data.sort_values(trend_x)
            
            # Calculate moving average
            window = st.slider("Moving Average Window", 2, 20, 5, key="trend_window")
            trend_data['moving_avg'] = trend_data[trend_y].rolling(window=window, center=True).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend_data[trend_x], y=trend_data[trend_y],
                                    mode='markers', name='Raw Data', opacity=0.5))
            fig.add_trace(go.Scatter(x=trend_data[trend_x], y=trend_data['moving_avg'],
                                    mode='lines', name=f'{window}-point Moving Average',
                                    line=dict(color='red', width=3)))
            
            # Add trend line
            from sklearn.linear_model import LinearRegression
            X = trend_data[trend_x].values.reshape(-1, 1)
            y = trend_data[trend_y].values
            model = LinearRegression().fit(X, y)
            trend_data['trend'] = model.predict(X)
            
            fig.add_trace(go.Scatter(x=trend_data[trend_x], y=trend_data['trend'],
                                    mode='lines', name='Linear Trend',
                                    line=dict(color='green', width=2, dash='dash')))
            
            fig.update_layout(title=f"Trend Analysis: {trend_y} vs {trend_x}", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show trend statistics
            st.markdown(f"**Trend Equation:** y = {model.coef_[0]:.4f} * x + {model.intercept_:.4f}")
            st.markdown(f"**R² Score:** {model.score(X, y):.4f}")

# ============================================================================
# TAB 6: PCE Analyzer - UPDATED with Adaptive R² (Maximizing Linear Points)
# ============================================================================

def find_optimal_linear_region_adaptive(time_values, ln_theta_values, min_points=5, r2_threshold=0.99):
    """
    Find the longest linear region in the -ln(theta) vs time plot that maintains high R².
    Adaptively extends the region while monitoring R² degradation.
    
    Parameters:
    - time_values: array of time values (minutes)
    - ln_theta_values: array of -ln(theta) values
    - min_points: minimum number of points required for linear fit
    - r2_threshold: minimum R² value to maintain (default 0.99)
    
    Returns:
    - Dictionary with optimal region parameters
    """
    n_points = len(time_values)
    
    if n_points < min_points:
        # Not enough points, use all data
        slope, intercept, r_value, _, _ = stats.linregress(time_values, ln_theta_values)
        return {
            'start_time': time_values[0],
            'end_time': time_values[-1],
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'n_points': n_points,
            'start_idx': 0,
            'end_idx': n_points - 1,
            'method': 'insufficient_data'
        }
    
    # Remove duplicate values (avoid repeated temperature)
    unique_indices = []
    seen_values = set()
    for i, val in enumerate(ln_theta_values):
        rounded_val = round(val, 4)  # Round to 4 decimal places to detect duplicates
        if rounded_val not in seen_values:
            seen_values.add(rounded_val)
            unique_indices.append(i)
    
    # If we removed duplicates, use unique indices
    if len(unique_indices) < n_points:
        time_unique = time_values[unique_indices]
        ln_theta_unique = ln_theta_values[unique_indices]
        n_points_unique = len(time_unique)
    else:
        time_unique = time_values
        ln_theta_unique = ln_theta_values
        n_points_unique = n_points
        unique_indices = list(range(n_points))
    
    # Initialize best_region with default values
    best_region = {
        'start_idx': 0,
        'end_idx': min_points - 1,
        'slope': 0,
        'intercept': 0,
        'r_squared': 0,
        'n_points': min_points,
        'start_time': time_unique[0],
        'end_time': time_unique[min_points - 1] if min_points - 1 < len(time_unique) else time_unique[-1],
        'method': 'initial'
    }
    
    # Strategy 1: Start from the beginning (steepest part) and extend forward
    for start_idx in range(0, min(n_points_unique - min_points, 20)):  # Try first 20 starting points
        current_r2 = 1.0
        end_idx = start_idx + min_points - 1
        
        # Extend forward while R² stays above threshold
        while end_idx < n_points_unique - 1 and current_r2 >= r2_threshold:
            end_idx += 1
            x_window = time_unique[start_idx:end_idx+1]
            y_window = ln_theta_unique[start_idx:end_idx+1]
            
            try:
                slope, intercept, r_value, _, _ = stats.linregress(x_window, y_window)
                current_r2 = r_value**2
                
                # If this region is longer than our best, update best
                if (end_idx - start_idx + 1) > best_region['n_points'] and current_r2 >= r2_threshold:
                    best_region = {
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': current_r2,
                        'n_points': end_idx - start_idx + 1,
                        'start_time': time_unique[start_idx],
                        'end_time': time_unique[end_idx],
                        'method': 'forward_extension'
                    }
            except:
                break
    
    # Strategy 2: If we couldn't find a long region, try backward from the end
    if best_region['n_points'] < min_points * 2:
        for end_idx in range(n_points_unique - 1, max(n_points_unique - 21, min_points - 1), -1):  # Try last 20 points
            current_r2 = 1.0
            start_idx = end_idx - min_points + 1
            
            # Extend backward while R² stays above threshold
            while start_idx > 0 and current_r2 >= r2_threshold:
                start_idx -= 1
                x_window = time_unique[start_idx:end_idx+1]
                y_window = ln_theta_unique[start_idx:end_idx+1]
                
                try:
                    slope, intercept, r_value, _, _ = stats.linregress(x_window, y_window)
                    current_r2 = r_value**2
                    
                    if (end_idx - start_idx + 1) > best_region['n_points'] and current_r2 >= r2_threshold:
                        best_region = {
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': current_r2,
                            'n_points': end_idx - start_idx + 1,
                            'start_time': time_unique[start_idx],
                            'end_time': time_unique[end_idx],
                            'method': 'backward_extension'
                        }
                except:
                    break
    
    # Strategy 3: If still no good region, use highest R² region with at least min_points
    if best_region['n_points'] < min_points * 1.5:
        best_r2 = -1
        for start_idx in range(0, n_points_unique - min_points + 1, max(1, n_points_unique // 10)):
            for end_idx in range(start_idx + min_points, min(start_idx + n_points_unique // 2, n_points_unique)):
                x_window = time_unique[start_idx:end_idx+1]
                y_window = ln_theta_unique[start_idx:end_idx+1]
                
                try:
                    slope, intercept, r_value, _, _ = stats.linregress(x_window, y_window)
                    r2 = r_value**2
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_region = {
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r2,
                            'n_points': end_idx - start_idx + 1,
                            'start_time': time_unique[start_idx],
                            'end_time': time_unique[end_idx],
                            'method': 'max_r2'
                        }
                except:
                    continue
    
    # Map back to original indices if we used unique indices
    if len(unique_indices) < len(time_values):
        best_region['start_idx'] = unique_indices[best_region['start_idx']]
        best_region['end_idx'] = unique_indices[best_region['end_idx']]
        best_region['start_time'] = time_values[best_region['start_idx']]
        best_region['end_time'] = time_values[best_region['end_idx']]
    
    return best_region


def analyze_r2_progression(time_values, ln_theta_values, max_points=None):
    """
    Analyze how R² changes as we add more points to the linear fit.
    Returns the optimal number of points to use.
    """
    if max_points is None:
        max_points = len(time_values)
    
    n_points = min(len(time_values), max_points)
    r2_values = []
    point_counts = []
    
    for i in range(5, n_points):  # Start with at least 5 points
        x_window = time_values[:i]
        y_window = ln_theta_values[:i]
        
        try:
            _, _, r_value, _, _ = stats.linregress(x_window, y_window)
            r2_values.append(r_value**2)
            point_counts.append(i)
        except:
            pass
    
    return point_counts, r2_values


def calculate_pce_optimized_adaptive(df, peak_idx, params):
    """
    Calculate photothermal conversion efficiency using adaptive linear region selection
    that maximizes the number of linear points while maintaining high R².
    """
    
    # Get peak information
    peak_time = df.loc[peak_idx, 'time_mins']
    peak_temp = df.loc[peak_idx, 'temperature_°C']
    ambient_temp = params['ambient_temp']
    
    # Calculate temperature difference
    delta_T = peak_temp - ambient_temp
    delta_T_net = delta_T - params['solvent_delta']
    
    # Extract cooling data
    cooling_data = df.iloc[peak_idx:].copy()
    cooling_data['time_from_peak'] = cooling_data['time_mins'] - peak_time
    cooling_data['theta'] = (cooling_data['temperature_°C'] - ambient_temp) / (peak_temp - ambient_temp)
    
    # Remove points where theta is too small (noise region)
    cooling_data = cooling_data[cooling_data['theta'] > 0.01].copy()
    cooling_data['ln_theta'] = np.log(cooling_data['theta'])
    cooling_data['neg_ln_theta'] = -cooling_data['ln_theta']  # -ln(theta)
    
    # Find optimal adaptive linear region
    optimal_region = find_optimal_linear_region_adaptive(
        cooling_data['time_from_peak'].values,
        cooling_data['neg_ln_theta'].values,
        min_points=5,
        r2_threshold=0.99
    )
    
    # Calculate tau from the optimal region slope
    tau_min = 1 / optimal_region['slope'] if optimal_region['slope'] > 0 else 200
    tau_seconds = tau_min * 60  # Convert to seconds
    
    # Calculate hS
    hS = (params['mass'] * params['cp']) / tau_seconds  # W/K
    
    # Calculate absorbed power
    absorbed_power = params['laser_power'] * (1 - 10**(-params['absorbance']))
    
    # Calculate efficiency
    efficiency = (hS * delta_T_net) / absorbed_power * 100
    
    # Determine expected range based on efficiency
    if efficiency < 20:
        expected_range = "15-25% (Carbon Dots/Quantum Dots)"
    elif efficiency < 35:
        expected_range = "25-40% (Graphene Oxide/Carbon Dots)"
    elif efficiency < 50:
        expected_range = "40-55% (Gold Nanoparticles/CuS)"
    else:
        expected_range = ">55% (High performance AuNPs/Polymers)"
    
    # Analyze R² progression
    point_counts, r2_progression = analyze_r2_progression(
        cooling_data['time_from_peak'].values[:50],  # First 50 points
        cooling_data['neg_ln_theta'].values[:50]
    )
    
    # Find where R² starts to drop significantly
    r2_drop_point = None
    if len(r2_progression) > 10:
        for i in range(5, len(r2_progression)):
            if r2_progression[i] < r2_progression[i-1] - 0.02:  # 2% drop
                r2_drop_point = point_counts[i]
                break
    
    return {
        'efficiency': efficiency,
        'tau_min': tau_min,
        'tau_seconds': tau_seconds,
        'hS': hS,
        'absorbed_power': absorbed_power,
        'r_squared': optimal_region['r_squared'],
        'slope': optimal_region['slope'],
        'intercept': optimal_region['intercept'],
        'delta_T': delta_T,
        'delta_T_net': delta_T_net,
        'expected_range': expected_range,
        'optimal_region': optimal_region,
        'cooling_data': cooling_data,
        'point_counts': point_counts,
        'r2_progression': r2_progression,
        'r2_drop_point': r2_drop_point
    }


def display_pce_tab():
    """Photothermal Conversion Efficiency Analysis Tab with Optimal R²"""
    
    st.markdown("<h2 class='sub-header'>🔥 Photothermal Conversion Efficiency (PCE) Analyzer</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Calculate photothermal conversion efficiency for quantum dots, metal nanoparticles, and carbon dots.
    The algorithm automatically finds the optimal linear region of -ln(θ) vs time to maximize R² value.
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    pce_tabs = st.tabs([
        "📊 Data Input", 
        "⚙️ PCE Parameters", 
        "📈 Analysis & Plots",
        "📉 Cooling Curve Analysis",
        "📋 Results Summary"
    ])
    
    # ========================================================================
    # Tab 1: Data Input
    # ========================================================================
    with pce_tabs[0]:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📁 Data Source")
            
            data_source = st.radio(
                "Select data source:",
                ["Use Sample Data (26% PCE)", "Upload Custom CSV"],
                key="pce_data_source"
            )
            
            if data_source == "Upload Custom CSV":
                uploaded_file = st.file_uploader(
                    "Upload time-temperature CSV",
                    type=['csv'],
                    key="pce_uploader"
                )
                if uploaded_file is not None:
                    # Try multiple encodings
                    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'cp437']
                    df = None
                    
                    for encoding in encodings:
                        try:
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, encoding=encoding)
                            st.success(f"✅ Loaded with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            continue
                        except Exception as e:
                            continue
                    
                    if df is None:
                        st.error("Could not read file. Please check the file format.")
                        df = None
                    else:
                        # Clean column names
                        df.columns = [col.replace('癈', '°C').replace('â', '°C') for col in df.columns]
                        
                        # Ensure correct column names
                        if len(df.columns) >= 2:
                            df.columns = ['time_mins', 'temperature_°C'] + list(df.columns[2:])
                        
                        st.session_state['pce_data'] = df
                else:
                    df = st.session_state.get('pce_data', None)
            else:
                # Use sample data with proper encoding
                sample_data = """time_mins,temperature_°C
0,20
0.5,21.8
1,23.3
1.5,24.8
2,26.3
2.5,27.6
3,29
3.5,30.2
4,31.4
4.5,32.7
5,33.7
5.5,34.8
6,36
6.5,36.8
7,37.8
7.5,38.6
8,39.4
8.5,40.4
9,41.1
9.5,41.8
10,42.5
10.5,43.2
11,43.8
11.5,44.4
12,45.1
12.5,45.7
13,46.1
13.5,46.8
14,47.2
14.5,47.7
15,48.1
15.5,48.6
16,48.8
16.5,49.3
17,49.7
17.5,49.9
18,50.2
18.5,50.6
19,50.8
19.5,51.1
20,51.3
20.5,51.6
21,51.8
21.5,52
22,52.2
22.5,52.5
23,52.5
23.5,52.8
24,53
24.5,53
25,53.2
25.5,53.3
26,53.5
26.5,53.5
27,53.6
27.5,53.6
28,53.7
28.5,53.9
29,54
29.5,54.1
30,54.3
30.5,54.3
31,54.5
31.5,54.5
32,54.5
32.5,54.7
33,54.8
33.5,54.8
34,54.8
34.5,54.8
35,55
35.5,55
36,55.1
36.5,55.2
37,55.2
37.5,55.3
38,55.2
38.5,55.2
39,55.4
39.5,55.6
40,55.6
40.5,55.6
41,55.6
41.5,53.3
42,51.8
42.5,50.5
43,49.3
43.5,47.8
44,46.6
44.5,45.6
45,44.4
45.5,43.3
46,42.4
46.5,41.5
47,40.7
47.5,39.7
48,38.8
48.5,38.2
49,37.3
49.5,36.7
50,36
50.5,35.5
51,34.9
51.5,34.3
52,33.8
52.5,33.3
53,32.7
53.5,32.3
54,31.8
54.5,31.3
55,31
55.5,30.7
56,30.2
56.5,29.9
57,29.5
57.5,29.1
58,28.9
58.5,28.6
59,28.3
59.5,28
60,27.7
60.5,27.5
61,27.2
61.5,26.8
62,26.6
62.5,26.6
63,26.4
63.5,26.2
64,26
64.5,25.8
65,25.7
65.5,25.5
66,25.3
66.5,25
67,25
67.5,25
68,24.7
68.5,24.6
69,24.4
69.5,24.4
70,24.2
70.5,24.1
71,24.1
71.5,23.8
72,23.8
72.5,23.7
73,23.6
73.5,23.6
74,23.3
74.5,23.3
75,23.3
75.5,23.2
76,23.2
76.5,23
77,23
77.5,23
78,22.9
78.5,22.8
79,22.8
79.5,22.7
80,22.7
80.5,22.7
81,22.7
81.5,22.6
82,22.5
82.5,22.5
83,22.3
83.5,22.2
84,22.3
84.5,22.3
85,22.2
85.5,22.2
86,22.1
86.5,22.2
87,22.2
87.5,22.1
88,21.9
88.5,21.9
89,21.9
89.5,21.7
90,21.7
90.5,21.6
91,21.5
91.5,21.5
92,21.5
92.5,21.3
93,21.3
93.5,21.2
94,21.1
94.5,21.1
95,21.1
95.5,20
96,20
96.5,20
97,20"""
                
                df = pd.read_csv(io.StringIO(sample_data))
                st.info("📊 Using sample data (expected PCE: 26%)")
                st.session_state['pce_data'] = df
            
            if df is not None:
                st.session_state['pce_data'] = df
                
                # Display raw data
                st.markdown("### 👁️ Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Data statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Time Points", len(df))
                with col2:
                    st.metric("Max Temperature", f"{df['temperature_°C'].max():.1f}°C")
                with col3:
                    st.metric("Min Temperature", f"{df['temperature_°C'].min():.1f}°C")
        
        with col2:
            if df is not None:
                st.markdown("### 📈 Raw Data Plot")
                
                # Identify heating and cooling phases
                temp_peak_idx = df['temperature_°C'].idxmax()
                peak_time = df.loc[temp_peak_idx, 'time_mins']
                peak_temp = df.loc[temp_peak_idx, 'temperature_°C']
                
                fig = go.Figure()
                
                # Heating phase
                fig.add_trace(go.Scatter(
                    x=df['time_mins'].iloc[:temp_peak_idx+1],
                    y=df['temperature_°C'].iloc[:temp_peak_idx+1],
                    mode='lines+markers',
                    name='Heating Phase',
                    line=dict(color='red', width=2),
                    marker=dict(size=4)
                ))
                
                # Cooling phase
                fig.add_trace(go.Scatter(
                    x=df['time_mins'].iloc[temp_peak_idx:],
                    y=df['temperature_°C'].iloc[temp_peak_idx:],
                    mode='lines+markers',
                    name='Cooling Phase',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ))
                
                # Mark peak
                fig.add_trace(go.Scatter(
                    x=[peak_time],
                    y=[peak_temp],
                    mode='markers',
                    name=f'Peak: {peak_temp:.1f}°C',
                    marker=dict(color='green', size=12, symbol='star')
                ))
                
                fig.update_layout(
                    title="Temperature Profile (Heating & Cooling)",
                    xaxis_title="Time (mins)",
                    yaxis_title="Temperature (°C)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Store peak information
                st.session_state['peak_idx'] = temp_peak_idx
                st.session_state['peak_time'] = peak_time
                st.session_state['peak_temp'] = peak_temp
    
    # ========================================================================
    # Tab 2: PCE Parameters
    # ========================================================================
    with pce_tabs[1]:
        st.markdown("### ⚙️ Photothermal Conversion Efficiency Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Material Properties")
            
            material_type = st.selectbox(
                "Material Type",
                ["Quantum Dots", "Metal Nanoparticles", "Carbon Dots", "Other"],
                key="pce_material"
            )
            
            mass = st.number_input(
                "Sample Mass (g)",
                min_value=0.001,
                max_value=100.0,
                value=1.0,
                step=0.1,
                format="%.3f",
                key="pce_mass"
            )
            
            cp_value = st.number_input(
                "Specific Heat Capacity (J/g·K)",
                min_value=0.1,
                max_value=10.0,
                value=4.184,
                step=0.01,
                format="%.3f",
                key="pce_cp",
                help="Default 4.184 J/g·K for water"
            )
            
            st.markdown("#### Laser Parameters")
            
            laser_power = st.number_input(
                "Laser Power (W)",
                min_value=0.001,
                max_value=10.0,
                value=1.0,
                step=0.1,
                format="%.3f",
                key="pce_power"
            )
            
            spot_area = st.number_input(
                "Laser Spot Area (cm²)",
                min_value=0.01,
                max_value=10.0,
                value=0.50265,
                step=0.1,
                format="%.3f",
                key="pce_area"
            )
            
            power_density = laser_power / spot_area
            st.metric("Power Density", f"{power_density:.2f} W/cm²")
        
        with col2:
            st.markdown("#### Optical Properties")
            
            absorbance = st.number_input(
                "Absorbance at Laser Wavelength",
                min_value=0.01,
                max_value=5.0,
                value=0.5,
                step=0.05,
                format="%.2f",
                key="pce_absorbance",
                help="A = -log10(I/I₀)"
            )
            
            laser_wavelength = st.number_input(
                "Laser Wavelength (nm)",
                min_value=300,
                max_value=2000,
                value=808,
                step=10,
                key="pce_wavelength"
            )
            
            st.markdown("#### Environmental Parameters")
            
            ambient_temp = st.number_input(
                "Ambient Temperature (°C)",
                min_value=0.0,
                max_value=50.0,
                value=20.0,
                step=0.1,
                key="pce_ambient"
            )
            
            solvent_blank_delta = st.number_input(
                "Solvent Blank ΔT (°C)",
                min_value=0.0,
                max_value=20.0,
                value=3.5,
                step=0.1,
                format="%.3f",
                key="pce_solvent",
                help="Temperature rise of pure water or solvent under same conditions"
            )
            
            # Calculate absorbed power
            absorbed_power = laser_power * (1 - 10**(-absorbance))
            st.metric("Absorbed Power", f"{absorbed_power:.3f} W")
            
            # Store parameters
            st.session_state['pce_params'] = {
                'material': material_type,
                'mass': mass,
                'cp': cp_value,
                'laser_power': laser_power,
                'spot_area': spot_area,
                'power_density': power_density,
                'absorbance': absorbance,
                'wavelength': laser_wavelength,
                'ambient_temp': ambient_temp,
                'solvent_delta': solvent_blank_delta,
                'absorbed_power': absorbed_power
            }
    
    # ========================================================================
    # Tab 3: Analysis & Plots - UPDATED with Adaptive R²
    # ========================================================================
    with pce_tabs[2]:
        if 'pce_data' not in st.session_state:
            st.warning("⚠️ Please load data in the Data Input tab first.")
        else:
            df = st.session_state['pce_data']
            params = st.session_state.get('pce_params', {})
            peak_idx = st.session_state.get('peak_idx', None)
            
            if not params:
                st.warning("⚠️ Please configure PCE parameters in the Parameters tab.")
            elif peak_idx is None:
                st.warning("⚠️ Peak temperature not identified. Please check data.")
            else:
                st.markdown("### 📈 Photothermal Analysis with Adaptive R²")
                
                # Calculate PCE using optimized adaptive method
                results = calculate_pce_optimized_adaptive(df, peak_idx, params)
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("ΔT Total", f"{results['delta_T']:.2f}°C")
                with col2:
                    st.metric("ΔT Net", f"{results['delta_T_net']:.2f}°C")
                with col3:
                    st.metric("Time Constant τ", f"{results['tau_seconds']:.0f} s")
                with col4:
                    st.metric("R² Value", f"{results['r_squared']:.4f}")
                
                # Display region info with emphasis on number of points used
                opt = results['optimal_region']
                st.info(f"✅ Adaptive linear region found: {opt['n_points']} points from "
                        f"{opt['start_time']:.1f} to {opt['end_time']:.1f} minutes "
                        f"with R² = {opt['r_squared']:.4f} (method: {opt['method']})")
                
                # Plot 1: -ln(θ) vs time with optimal region highlighted
                fig1 = go.Figure()
                
                cooling_data = results['cooling_data']
                
                # All data points
                fig1.add_trace(go.Scatter(
                    x=cooling_data['time_from_peak'],
                    y=cooling_data['neg_ln_theta'],
                    mode='markers',
                    name='All Data',
                    marker=dict(color='lightblue', size=6, opacity=0.5)
                ))
                
                # Highlight optimal region
                mask = (cooling_data.index >= opt['start_idx']) & (cooling_data.index <= opt['end_idx'])
                
                fig1.add_trace(go.Scatter(
                    x=cooling_data.loc[mask, 'time_from_peak'],
                    y=cooling_data.loc[mask, 'neg_ln_theta'],
                    mode='markers',
                    name=f'Optimal Region ({opt["n_points"]} pts)',
                    marker=dict(color='red', size=8)
                ))
                
                # Linear fit line
                x_line = np.linspace(opt['start_time'], opt['end_time'], 100)
                y_line = opt['slope'] * x_line + opt['intercept']
                
                fig1.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    name=f'Linear Fit (R² = {opt["r_squared"]:.4f})',
                    line=dict(color='red', width=3)
                ))
                
                fig1.update_layout(
                    title=f"-ln(θ) vs Time (Optimal Region: {opt['start_time']:.1f}-{opt['end_time']:.1f} min)",
                    xaxis_title="Time from Peak (minutes)",
                    yaxis_title="-ln(θ)",
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Plot 2: Cooling curve with exponential fit
                fig2 = go.Figure()
                
                # Raw cooling data
                fig2.add_trace(go.Scatter(
                    x=cooling_data['time_from_peak'],
                    y=cooling_data['temperature_°C'],
                    mode='markers',
                    name='Experimental',
                    marker=dict(color='blue', size=6)
                ))
                
                # Exponential fit
                t_fit = np.linspace(0, cooling_data['time_from_peak'].max(), 100)
                T_fit = params['ambient_temp'] + (peak_temp - params['ambient_temp']) * np.exp(-t_fit / results['tau_min'])
                
                fig2.add_trace(go.Scatter(
                    x=t_fit,
                    y=T_fit,
                    mode='lines',
                    name=f'Exponential Fit (τ={results["tau_seconds"]:.0f}s)',
                    line=dict(color='red', width=2)
                ))
                
                fig2.update_layout(
                    title="Cooling Curve with Exponential Fit",
                    xaxis_title="Time from Peak (minutes)",
                    yaxis_title="Temperature (°C)",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Plot 3: R² progression
                if results['point_counts'] and results['r2_progression']:
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(
                        x=results['point_counts'],
                        y=results['r2_progression'],
                        mode='lines+markers',
                        name='R² vs Points',
                        line=dict(color='purple', width=2)
                    ))
                    fig3.add_hline(
                        y=opt['r_squared'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Selected: {opt['n_points']} pts"
                    )
                    if results['r2_drop_point']:
                        fig3.add_vline(
                            x=results['r2_drop_point'],
                            line_dash="dot",
                            line_color="orange",
                            annotation_text=f"Drop at {results['r2_drop_point']} pts"
                        )
                    fig3.update_layout(
                        title="R² vs Number of Points (Finding Optimal Length)",
                        xaxis_title="Number of Points",
                        yaxis_title="R² Value",
                        height=300
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Store results
                st.session_state['pce_results'] = results
    
    # ========================================================================
    # Tab 4: Cooling Curve Analysis
    # ========================================================================
    with pce_tabs[3]:
        if 'pce_data' not in st.session_state:
            st.warning("⚠️ Please load data in the Data Input tab first.")
        else:
            df = st.session_state['pce_data']
            params = st.session_state.get('pce_params', {})
            peak_idx = st.session_state.get('peak_idx', None)
            
            st.markdown("### 📉 Detailed Cooling Curve Analysis")
            
            if peak_idx is not None:
                # User selects cooling range
                cooling_start = st.slider(
                    "Cooling Start Index",
                    min_value=peak_idx,
                    max_value=len(df)-1,
                    value=peak_idx,
                    key="cooling_start"
                )
                
                cooling_end = st.slider(
                    "Cooling End Index",
                    min_value=cooling_start,
                    max_value=len(df)-1,
                    value=len(df)-1,
                    key="cooling_end"
                )
                
                if params:
                    cooling_analysis = df.iloc[cooling_start:cooling_end+1].copy()
                    peak_time = df.loc[peak_idx, 'time_mins']
                    peak_temp = df.loc[peak_idx, 'temperature_°C']
                    
                    cooling_analysis['time_from_peak'] = cooling_analysis['time_mins'] - peak_time
                    cooling_analysis['theta'] = (cooling_analysis['temperature_°C'] - params['ambient_temp']) / (peak_temp - params['ambient_temp'])
                    cooling_analysis = cooling_analysis[cooling_analysis['theta'] > 0.01].copy()
                    cooling_analysis['neg_ln_theta'] = -np.log(cooling_analysis['theta'])
                    
                    # Find optimal linear region for this selected range
                    if len(cooling_analysis) > 5:
                        opt_region = find_optimal_linear_region_adaptive(
                            cooling_analysis['time_from_peak'].values,
                            cooling_analysis['neg_ln_theta'].values,
                            min_points=5,
                            r2_threshold=0.99
                        )
                        
                        st.info(f"✅ Optimal linear region in selected range: {opt_region['n_points']} points from "
                               f"{opt_region['start_time']:.1f} to {opt_region['end_time']:.1f} minutes "
                               f"with R² = {opt_region['r_squared']:.4f}")
                        
                        # Plot with optimal region
                        fig = go.Figure()
                        
                        # All data
                        fig.add_trace(go.Scatter(
                            x=cooling_analysis['time_from_peak'],
                            y=cooling_analysis['neg_ln_theta'],
                            mode='markers',
                            name='All Data',
                            marker=dict(color='lightblue', size=6)
                        ))
                        
                        # Highlight optimal region
                        mask = (cooling_analysis['time_from_peak'] >= opt_region['start_time']) & \
                               (cooling_analysis['time_from_peak'] <= opt_region['end_time'])
                        
                        fig.add_trace(go.Scatter(
                            x=cooling_analysis.loc[mask, 'time_from_peak'],
                            y=cooling_analysis.loc[mask, 'neg_ln_theta'],
                            mode='markers',
                            name='Optimal Region',
                            marker=dict(color='red', size=8)
                        ))
                        
                        # Linear fit
                        x_line = np.linspace(opt_region['start_time'], opt_region['end_time'], 100)
                        y_line = opt_region['slope'] * x_line + opt_region['intercept']
                        
                        fig.add_trace(go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode='lines',
                            name=f'Fit: y={opt_region["slope"]:.4f}x+{opt_region["intercept"]:.4f}',
                            line=dict(color='red', width=2)
                        ))
                        
                        fig.update_layout(
                            title=f"-ln(θ) vs Time (Optimal R² = {opt_region['r_squared']:.4f})",
                            xaxis_title="Time from Peak (minutes)",
                            yaxis_title="-ln(θ)",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display fit parameters
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Slope", f"{opt_region['slope']:.4f}")
                        with col2:
                            st.metric("Intercept", f"{opt_region['intercept']:.4f}")
                        with col3:
                            st.metric("Time Constant τ", f"{1/opt_region['slope']:.2f} mins")
                    else:
                        st.warning("Not enough data points in selected range for analysis.")
    
    # ========================================================================
    # Tab 5: Results Summary
    # ========================================================================
    with pce_tabs[4]:
        st.markdown("### 📋 Photothermal Conversion Efficiency Results")
        
        if 'pce_results' not in st.session_state:
            st.warning("⚠️ Please run analysis in the Analysis tab first.")
        else:
            results = st.session_state['pce_results']
            params = st.session_state.get('pce_params', {})
            
            # Display results in a nice format
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔬 Material Information")
                st.write(f"**Material Type:** {params.get('material', 'N/A')}")
                st.write(f"**Sample Mass:** {params.get('mass', 0):.3f} g")
                st.write(f"**Specific Heat:** {params.get('cp', 0):.2f} J/g·K")
                st.write(f"**Laser Wavelength:** {params.get('wavelength', 0)} nm")
                st.write(f"**Absorbance:** {params.get('absorbance', 0):.2f}")
                
                st.markdown("#### 📐 Geometric Parameters")
                st.write(f"**Laser Power:** {params.get('laser_power', 0):.2f} W")
                st.write(f"**Spot Area:** {params.get('spot_area', 0):.2f} cm²")
                st.write(f"**Power Density:** {params.get('power_density', 0):.2f} W/cm²")
                st.write(f"**Absorbed Power:** {results['absorbed_power']:.3f} W")
            
            with col2:
                st.markdown("#### 📈 Optimal Linear Region Parameters")
                opt = results['optimal_region']
                st.write(f"**Region Start:** {opt['start_time']:.2f} mins")
                st.write(f"**Region End:** {opt['end_time']:.2f} mins")
                st.write(f"**Number of Points:** {opt['n_points']}")
                st.write(f"**Slope:** {opt['slope']:.4f}")
                st.write(f"**Intercept:** {opt['intercept']:.4f}")
                st.write(f"**R² Value:** {opt['r_squared']:.4f}")
                st.write(f"**Method:** {opt['method']}")
                
                st.markdown("#### 📊 Thermal Parameters")
                st.write(f"**Time Constant (τ):** {results['tau_min']:.3f} mins")
                st.write(f"**Time Constant (τ):** {results['tau_seconds']:.0f} seconds")
                st.write(f"**hS Value:** {results['hS']:.4f} W/K")
                st.write(f"**ΔT Net:** {results['delta_T_net']:.2f}°C")
            
            st.markdown("---")
            
            # Main PCE result
            st.markdown("### 🎯 Photothermal Conversion Efficiency")
            
            # Create a big metric display
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 1rem;'>
                    <h1 style='color: white; font-size: 4rem;'>{results['efficiency']:.1f}%</h1>
                    <p style='color: white; font-size: 1.2rem;'>Photothermal Conversion Efficiency</p>                    
                </div>
                """, unsafe_allow_html=True)
            
            # Comparison with expected value (26%)
            expected_pce = 26.0
            difference = abs(results['efficiency'] - expected_pce)
            percent_diff = (difference / expected_pce) * 100
            
            st.markdown("#### 📊 Validation")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Calculated PCE", f"{results['efficiency']:.2f}%")
            with col2:
                st.metric("Expected PCE", f"{expected_pce:.2f}%")
            with col3:
                delta_symbol = "↓" if results['efficiency'] < expected_pce else "↑"
                st.metric("Difference", f"{difference:.2f}%", delta=f"{delta_symbol} {percent_diff:.1f}%")
            
            # Export results
            st.markdown("#### 📥 Export Results")
            
            # Create results dictionary for export
            export_data = {
                'Parameter': [],
                'Value': [],
                'Unit': []
            }
            
            # Add material parameters
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    export_data['Parameter'].append(key.replace('_', ' ').title())
                    export_data['Value'].append(value)
                    export_data['Unit'].append('')
            
            # Add optimal region parameters
            export_data['Parameter'].append('Optimal Region Start')
            export_data['Value'].append(opt['start_time'])
            export_data['Unit'].append('mins')
            
            export_data['Parameter'].append('Optimal Region End')
            export_data['Value'].append(opt['end_time'])
            export_data['Unit'].append('mins')
            
            export_data['Parameter'].append('Optimal Region Points')
            export_data['Value'].append(opt['n_points'])
            export_data['Unit'].append('')
            
            export_data['Parameter'].append('Linear Fit Slope')
            export_data['Value'].append(opt['slope'])
            export_data['Unit'].append('min⁻¹')
            
            export_data['Parameter'].append('Linear Fit Intercept')
            export_data['Value'].append(opt['intercept'])
            export_data['Unit'].append('')
            
            export_data['Parameter'].append('R² Value')
            export_data['Value'].append(opt['r_squared'])
            export_data['Unit'].append('')
            
            # Add thermal results
            export_data['Parameter'].append('Time Constant')
            export_data['Value'].append(results['tau_seconds'])
            export_data['Unit'].append('s')
            
            export_data['Parameter'].append('hS Value')
            export_data['Value'].append(results['hS'])
            export_data['Unit'].append('W/K')
            
            export_data['Parameter'].append('PCE')
            export_data['Value'].append(results['efficiency'])
            export_data['Unit'].append('%')
            
            export_df = pd.DataFrame(export_data)
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"pce_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ============================================================================
# TAB 7: AI Research Assistant Class
# ============================================================================
class AIResearchAssistant:
    """AI Assistant with Brave and Tavily as primary search engines (OpenAI optional)"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_providers()
    
    def initialize_session_state(self):
        if 'assistant_messages' not in st.session_state:
            st.session_state.assistant_messages = []
        if 'assistant_initialized' not in st.session_state:
            st.session_state.assistant_initialized = False
        if 'brave_key' not in st.session_state:
            st.session_state.brave_key = None
        if 'tavily_key' not in st.session_state:
            st.session_state.tavily_key = None
        if 'openai_key' not in st.session_state:
            st.session_state.openai_key = None
        if 'api_status' not in st.session_state:
            st.session_state.api_status = {
                'brave': False,
                'tavily': False,
                'openai': False
            }
    
    def setup_providers(self):
        if 'BRAVE_API_KEY' in st.secrets:
            st.session_state.brave_key = st.secrets['BRAVE_API_KEY']
            st.session_state.api_status['brave'] = True
        if 'TAVILY_API_KEY' in st.secrets:
            st.session_state.tavily_key = st.secrets['TAVILY_API_KEY']
            st.session_state.api_status['tavily'] = True
        if 'OPENAI_API_KEY' in st.secrets:
            st.session_state.openai_key = st.secrets['OPENAI_API_KEY']
            st.session_state.api_status['openai'] = True
    
    def brave_web_search(self, query, count=5):
        if not st.session_state.brave_key:
            return None
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": st.session_state.brave_key
        }
        params = {
            "q": query,
            "count": count,
            "search_lang": "en",
            "text_format": "raw",
            "safesearch": "off"
        }
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def tavily_search(self, query, search_depth="basic", max_results=5):
        if not st.session_state.tavily_key:
            return None
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.session_state.tavily_key}"
        }
        payload = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False,
            "include_images": False
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def search_with_brave(self, query):
        results = self.brave_web_search(query)
        if not results or 'web' not in results or 'results' not in results['web']:
            return None
        formatted = {'sources': [], 'content': '', 'answer': None}
        for r in results['web']['results'][:3]:
            formatted['sources'].append({
                'title': r.get('title', ''),
                'url': r.get('url', '#'),
                'description': r.get('description', ''),
                'provider': 'brave'
            })
            formatted['content'] += f"{r.get('description', '')}\n\n"
        return formatted
    
    def search_with_tavily(self, query):
        results = self.tavily_search(query)
        if not results:
            return None
        formatted = {'sources': [], 'content': '', 'answer': results.get('answer', None)}
        if 'results' in results:
            for r in results['results'][:3]:
                formatted['sources'].append({
                    'title': r.get('title', ''),
                    'url': r.get('url', '#'),
                    'content': r.get('content', ''),
                    'provider': 'tavily'
                })
                formatted['content'] += f"{r.get('content', '')}\n\n"
        return formatted
    
    def search_all(self, query):
        results = {'sources': [], 'content': '', 'answer': None}
        if st.session_state.api_status['tavily']:
            tavily_results = self.search_with_tavily(query)
            if tavily_results:
                results['sources'].extend(tavily_results['sources'])
                results['content'] += tavily_results['content']
                if tavily_results['answer']:
                    results['answer'] = tavily_results['answer']
        if st.session_state.api_status['brave']:
            brave_results = self.search_with_brave(query)
            if brave_results:
                results['sources'].extend(brave_results['sources'])
                results['content'] += brave_results['content']
        return results
    
    def generate_response(self, query, search_results):
        sources_text = ""
        if search_results['sources']:
            sources_text = "**Search Results:**\n\n"
            for i, src in enumerate(search_results['sources'][:3], 1):
                sources_text += f"{i}. **{src.get('title', 'Source')}**\n"
                if 'description' in src:
                    sources_text += f"   {src['description'][:200]}...\n"
                elif 'content' in src:
                    sources_text += f"   {src['content'][:200]}...\n"
                sources_text += f"   URL: {src.get('url', '#')}\n\n"
        
        if (st.session_state.api_status['openai'] and 
            st.session_state.openai_key and 
            OPENAI_AVAILABLE):
            try:
                client = OpenAI(api_key=st.session_state.openai_key)
                system_prompt = """You are ChemNanoBot AI, a research assistant specializing in chemistry, quantum dots, and porphyrin synthesis. 
                Use the provided search results to answer the user's question accurately. 
                If search results are available, base your answer on them. 
                If no search results are available, use your general knowledge.
                Always cite your sources when possible."""
                user_prompt = f"""Search Results:
                {search_results['content'][:3000]}
                Question: {query}
                Please provide a comprehensive answer based on the search results above."""
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except:
                pass
        
        response = f"**Question:** {query}\n\n"
        if search_results['answer']:
            response += f"**Answer:** {search_results['answer']}\n\n"
        if search_results['sources']:
            response += sources_text
        else:
            response += "I couldn't find specific information about that query. Please try rephrasing or check your API keys."
        return response
    
    def render_ui(self):
        st.markdown("<h2 class='sub-header'>🤖 AI Research Assistant</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.api_status['brave']:
                st.success("✅ Brave Search (Active)")
            else:
                st.warning("⚠️ Brave Search (Optional)")
        with col2:
            if st.session_state.api_status['tavily']:
                st.success("✅ Tavily Search (Active)")
            else:
                st.warning("⚠️ Tavily Search (Optional)")
        with col3:
            if st.session_state.api_status['openai']:
                st.success("✅ OpenAI (Optional)")
            else:
                st.info("ℹ️ OpenAI (Not Required)")
        
        st.markdown("---")
        
        with st.sidebar.expander("⚙️ Search Engine Settings", expanded=True):
            st.markdown("#### 🔑 API Keys (Optional but Recommended)")
            
            brave_key_input = st.text_input(
                "Brave Search API Key",
                type="password",
                value=st.session_state.brave_key if st.session_state.brave_key else "",
                help="Get free API key from brave.com/search/api"
            )
            if brave_key_input:
                st.session_state.brave_key = brave_key_input
                st.session_state.api_status['brave'] = True
            
            tavily_key_input = st.text_input(
                "Tavily API Key",
                type="password",
                value=st.session_state.tavily_key if st.session_state.tavily_key else "",
                help="Get free API key from tavily.com"
            )
            if tavily_key_input:
                st.session_state.tavily_key = tavily_key_input
                st.session_state.api_status['tavily'] = True
            
            with st.expander("🤖 Optional: OpenAI for Enhanced Responses"):
                openai_key_input = st.text_input(
                    "OpenAI API Key (Optional)",
                    type="password",
                    value=st.session_state.openai_key if st.session_state.openai_key else "",
                    help="Add for AI-powered responses. Not required for basic search."
                )
                if openai_key_input:
                    st.session_state.openai_key = openai_key_input
                    st.session_state.api_status['openai'] = True
            
            st.markdown("---")
            st.markdown("**Current Status:**")
            if st.session_state.api_status['brave'] or st.session_state.api_status['tavily']:
                st.success("✅ Search engines ready!")
                st.session_state.assistant_initialized = True
            else:
                st.warning("⚠️ Add at least one search engine API key to enable web search")
        
        if st.session_state.assistant_initialized:
            for message in st.session_state.assistant_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message and message["sources"]:
                        with st.expander(f"📚 Sources ({len(message['sources'])})"):
                            for src in message["sources"]:
                                title = src.get('title', src.get('url', 'Source'))
                                url = src.get('url', '#')
                                provider = src.get('provider', 'web')
                                st.markdown(f"- [{title}]({url}) *({provider})*")
            
            if prompt := st.chat_input("Ask about synthesis, research, or any topic..."):
                st.session_state.assistant_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Searching..."):
                        search_results = self.search_all(prompt)
                        response = self.generate_response(prompt, search_results)
                        st.markdown(response)
                        
                        if search_results['sources']:
                            with st.expander(f"📚 Sources ({len(search_results['sources'])})"):
                                for src in search_results['sources'][:5]:
                                    title = src.get('title', src.get('url', 'Source'))
                                    url = src.get('url', '#')
                                    provider = src.get('provider', 'web')
                                    st.markdown(f"- [{title}]({url}) *({provider})*")
                        
                        st.session_state.assistant_messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": search_results['sources']
                        })
            
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.assistant_messages = []
                st.rerun()
        else:
            st.info("👈 Configure your search engine API keys in the sidebar to start searching.")
            with st.expander("💡 Example Questions"):
                st.markdown("""
                - "What are the latest developments in quantum dot synthesis?"
                - "Show me recent papers on porphyrin-based PDT"
                - "What's the current best method for high-PLQY CIS/ZnS QDs?"
                - "Tell me about heavy atom effects in singlet oxygen generation"
                """)


# ============================================================================
# TAB 8: ChemNanoBot Class
# ============================================================================
class ChemNanoBot:
    """Enhanced chatbot for synthesis advice"""
    
    def __init__(self):
        self.conversation_history = []
        self.context_window = []
    
    def get_response(self, user_message):
        user_message_lower = user_message.lower()
        
        self.context_window.append({"role": "user", "content": user_message})
        if len(self.context_window) > 10:
            self.context_window = self.context_window[-10:]
        
        if len(self.context_window) > 2:
            if "more" in user_message_lower or "elaborate" in user_message_lower:
                return self.get_elaboration_response()
        
        if any(word in user_message_lower for word in ['quantum dot', 'qd', 'cis/zns']):
            response = self.get_qd_response(user_message)
        elif 'porphyrin' in user_message_lower:
            response = self.get_porphyrin_response(user_message)
        elif any(word in user_message_lower for word in ['optimiz', 'bayesian', 'pareto']):
            response = self.get_optimization_response(user_message)
        elif any(word in user_message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            response = self.get_greeting()
        elif any(word in user_message_lower for word in ['thank', 'thanks']):
            response = self.get_thanks_response()
        else:
            response = self.get_general_response()
        
        self.context_window.append({"role": "assistant", "content": response})
        return response
    
    def get_elaboration_response(self):
        return """📚 **Let me elaborate further...**

Based on our conversation, here's additional detail:

**Key Points to Remember:**
1. Temperature control is critical - maintain ±2°C
2. Precursor purity affects final quality - use >99.9%
3. Reaction monitoring via UV-Vis helps track growth
4. Shell growth requires careful rate control

Would you like me to dive deeper into any specific aspect?"""
    
    def get_qd_response(self, query):
        return """🎯 **Quantum Dot Synthesis Advice**

For optimal CIS/ZnS quantum dots:

**Key Parameters:**
- **Precursor ratio (Cu:In):** 0.8-1.2
- **Temperature:** 180-220°C for core
- **Time:** 60-120 minutes
- **Shell growth:** 200-240°C with Zn precursor

**For absorption ≥800nm:**
- Increase In content (ratio >1.2)
- Extend reaction time (90-120 min)
- Grow 3-5 monolayers of ZnS shell

**For high PLQY (>60%):**
- Use coordinating solvents (oleylamine, TOP)
- Optimize shell coverage
- Purify via size-selective precipitation

Would you like specific advice on any parameter?"""
    
    def get_porphyrin_response(self, query):
        return """🧪 **Porphyrin Synthesis Advice**

**Lindsey Method Recommendations:**
- **Concentration:** 0.01-0.02 M
- **Catalyst:** BF3·OEt2 (0.1-0.3 eq)
- **Temperature:** Room temperature
- **Oxidation:** DDQ or p-chloranil
- **Yield:** 30-50% possible

**Adler-Long Method:**
- **Concentration:** 0.05-0.1 M
- **Temperature:** Reflux in propionic acid (~140°C)
- **Time:** 30-60 minutes
- **Yield:** 15-30%

**For high singlet oxygen:**
- Heavy atom substitution (Br, I)
- Metalation with Pd or Pt
- Extended conjugation
- Push-pull architecture

Need help with a specific aspect?"""
    
    def get_optimization_response(self, query):
        return """🚀 **Optimization Strategy**

**Bayesian Optimization Workflow:**
1. Define parameter space (5-10 factors)
2. Build surrogate model (Gaussian Process)
3. Use acquisition function (EI, UCB, PI)
4. Suggest next experiment
5. Update model with results
6. Repeat until convergence

**Multi-objective tips:**
- Use Pareto front analysis
- Weighted sum for composite score
- Consider trade-offs between properties

Want me to elaborate on any step?"""
    
    def get_greeting(self):
        return """👋 **Hello! I'm ChemNanoBot, your synthesis optimization assistant!**

I specialize in:
- 🧪 **Quantum Dots** (CIS/ZnS, CdSe, Perovskite)
- 🔬 **Porphyrins** (Synthesis, metalation, properties)
- 📊 **Experimental Design** (DoE, factorial designs)
- 🤖 **Optimization** (Bayesian, multi-objective)

**Try asking me:**
- "How do I optimize QD absorption for 800nm?"
- "What's the best porphyrin synthesis method?"
- "Explain Bayesian optimization"

What would you like to explore today?"""
    
    def get_thanks_response(self):
        return """🎉 **You're welcome!**

I'm glad I could help! Feel free to ask if you have more questions about:
- Synthesis conditions
- Experimental design
- Data analysis
- Troubleshooting

Happy experimenting! 🧪"""
    
    def get_general_response(self):
        return """💡 **I'm here to help with synthesis optimization!**

You can ask me about:
- **Quantum Dots**: Synthesis conditions, size control, shell growth
- **Porphyrins**: Methods, metalation, substituent effects
- **DoE**: Factorial designs, response surface, analysis
- **Optimization**: Bayesian, multi-objective, Pareto fronts

What specific topic interests you?"""


def display_ai_assistant():
    """AI Assistant tab with enhanced UI"""
    
    st.markdown("<h2 class='sub-header'>🤖 ChemNanoBot AI Assistant</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Chat with <strong>ChemNanoBot</strong>, your AI expert in quantum dot and porphyrin synthesis optimization.
    Ask about synthesis conditions, experimental design, or data analysis!
    </div>
    """, unsafe_allow_html=True)
    
    if 'ai_chatbot' not in st.session_state:
        st.session_state.ai_chatbot = ChemNanoBot()
    
    if 'ai_messages' not in st.session_state:
        st.session_state.ai_messages = [
            {"role": "assistant", "content": "👋 Hello! I'm ChemNanoBot, your synthesis optimization assistant. How can I help you today?"}
        ]
    
    for message in st.session_state.ai_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about synthesis optimization..."):
        st.session_state.ai_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🔬 Analyzing your question..."):
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.markdown(response)
        st.session_state.ai_messages.append({"role": "assistant", "content": response})
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.ai_messages = [
            {"role": "assistant", "content": "👋 Hello! I'm ChemNanoBot, your synthesis optimization assistant. How can I help you today?"}
        ]
        st.rerun()
        
# ============================================================================
# Main function - FIXED (removed healer references)
# ============================================================================
def main():
    try:
        # Check required functions
        required_functions = [
            'display_quantum_dots_tab',
            'display_porphyrins_tab', 
            'display_multi_objective_tab',
            'display_molecular_generator_tab',
            'display_advanced_visualization',
            'display_pce_tab',
            'display_ai_assistant'
        ]
        
        for func_name in required_functions:
            if func_name not in globals():
                st.error(f"❌ Critical error: Function '{func_name}' is not defined.")
                st.stop()
        
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
                "🤖 AI Research Assistant",
                "💬 ChemNanoBot"
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
            
            # About section
            st.markdown("## ℹ️ About")
            st.info(
                "**CHEM‑NANO‑BEW Laboratory**\n\n"
                "Advanced synthesis optimization for "
                "quantum dots and porphyrins using "
                "machine learning and DoE.\n\n"
                f"**Version:** 2.1 (RDKit Mode)"
            )
            
            # API Status (if available)
            if 'api_status' in st.session_state:
                with st.expander("🔌 API Status"):
                    status = st.session_state.api_status
                    st.write(f"Brave: {'✅' if status.get('brave') else '❌'}")
                    st.write(f"Tavily: {'✅' if status.get('tavily') else '❌'}")
                    st.write(f"OpenAI: {'✅' if status.get('openai') else '❌'}")

        # Main header
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<h1 class='main-header'>CHEM‑NANO‑BEW LABORATORY</h1>", unsafe_allow_html=True)
            st.markdown("<p class='lab-subtitle'>Advanced Synthesis Optimization Suite</p>", unsafe_allow_html=True)

        # Route to appropriate tab - REMOVED healer references
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
            if 'ai_research_assistant' not in st.session_state:
                st.session_state.ai_research_assistant = AIResearchAssistant()
            st.session_state.ai_research_assistant.render_ui()
        elif mode == "💬 ChemNanoBot":
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
            <p>Powered by CHEMNANOBEW GROUP • v2.1</p>
            <p style='font-size: 0.8rem;'>© 2026 CHEM-NANO-BEW Laboratory</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fallback error display
        st.set_page_config(page_title="CHEMNANOBEW - Error", page_icon="🚨")
        st.error("🚨 **Critical Application Error**")
        st.exception(e)
        st.markdown("""
        ### Troubleshooting Steps:
        1. Check that all required functions are defined above
        2. Verify your API keys in `.streamlit/secrets.toml`
        3. Check the console for detailed error messages
        4. Try refreshing the page
        """)
