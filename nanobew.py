"""
chemnanobew_app.py – Synthesis Optimization Suite (RDKit‑Mode)
Deploy on Streamlit Cloud with no RDKit dependency.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.linear_model import LinearRegression
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
# [ALL YOUR TAB FUNCTIONS GO HERE - display_quantum_dots_tab, 
#  display_porphyrins_tab, display_multi_objective_tab, 
#  display_molecular_generator_tab, display_advanced_visualization,
#  display_pce_tab, AIResearchAssistant, ChemNanoBot, display_ai_assistant]
# ============================================================================

# [PASTE ALL YOUR TAB FUNCTIONS HERE - they are too long to include in this response]
# The functions from your original code should be pasted here

# ============================================================================
# Main function
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
            'display_ai_assistant'
        ]
        
        for func_name in required_functions:
            if func_name not in globals():
                st.error(f"❌ Critical error: Function '{func_name}' is not defined.")
                st.stop()
        
        with st.sidebar:
            if os.path.exists("images") and os.listdir("images"):
                st.image(os.path.join("images", os.listdir("images")[0]), use_column_width=True)
            else:
                st.markdown("""
                <div class='sidebar-logo'>
                    <div style='font-size:3rem;'>🧪</div>
                    <div class='sidebar-logo-text'>CHEM‑NANO‑BEW</div>
                    <div style='color:#ecf0f1; font-size:0.9rem;'>LABORATORY</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
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
            
            with st.expander("📸 Upload Logo"):
                logo = st.file_uploader("Choose image", type=['png','jpg','jpeg','gif'], key="logo_uploader")
                if logo is not None:
                    saved_path = save_uploaded_image(logo)
                    if saved_path:
                        st.success("✅ Logo uploaded!")
                        st.rerun()
            
            st.markdown("---")
            st.markdown("## 📁 Data Management")
            uploaded_file = st.file_uploader("Upload CSV data", type=['csv'], key="data_uploader")
            
            if uploaded_file is not None:
                st.success(f"✅ Loaded: {uploaded_file.name}")
            
            st.markdown("---")
            st.markdown("## ℹ️ About")
            st.info(
                "**CHEM‑NANO‑BEW Laboratory**\n\n"
                "Advanced synthesis optimization for "
                "quantum dots and porphyrins using "
                "machine learning and DoE.\n\n"
                f"**Version:** 2.1 (RDKit Mode)"
            )
            
            if 'api_status' in st.session_state:
                with st.expander("🔌 API Status"):
                    status = st.session_state.api_status
                    st.write(f"Brave: {'✅' if status.get('brave') else '❌'}")
                    st.write(f"Tavily: {'✅' if status.get('tavily') else '❌'}")
                    st.write(f"OpenAI: {'✅' if status.get('openai') else '❌'}")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<h1 class='main-header'>CHEM‑NANO‑BEW LABORATORY</h1>", unsafe_allow_html=True)
            st.markdown("<p class='lab-subtitle'>Advanced Synthesis Optimization Suite</p>", unsafe_allow_html=True)

        if mode == "🧪 Quantum Dots":
            healer.healing_decorator(display_quantum_dots_tab)(uploaded_file)
        elif mode == "🔬 Porphyrins":
            healer.healing_decorator(display_porphyrins_tab)(uploaded_file)
        elif mode == "🎯 Multi-Objective":
            healer.healing_decorator(display_multi_objective_tab)()
        elif mode == "🧬 Molecular Generator":
            healer.healing_decorator(display_molecular_generator_tab)()
        elif mode == "📊 Advanced Visualization":
            healer.healing_decorator(display_advanced_visualization)(uploaded_file)
        elif mode == "🔥 PCE Analyzer":
            healer.healing_decorator(display_pce_tab)()
        elif mode == "🤖 AI Research Assistant":
            if 'ai_research_assistant' not in st.session_state:
                st.session_state.ai_research_assistant = AIResearchAssistant()
            st.session_state.ai_research_assistant.render_ui()
        elif mode == "💬 ChemNanoBot":
            healer.healing_decorator(display_ai_assistant)()
        else:
            st.error(f"Unknown mode selected: {mode}")
            
    except Exception as e:
        st.error(f"⚠️ An error occurred:")
        st.exception(e)
        st.info("Please check the console logs or refresh the page.")
        if 'healer' in globals():
            st.info("🔄 Attempting self-healing...")
            healer.heal_from_error(main, e)

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
        st.set_page_config(page_title="CHEMNANOBEW - Error", page_icon="🚨")
        st.error("🚨 **Critical Application Error**")
        st.exception(e)
        st.markdown("""
        ### Troubleshooting Steps:
        1. Check that all required functions are defined
        2. Verify your API keys in `.streamlit/secrets.toml`
        3. Check the console for detailed error messages
        4. Try refreshing the page
        """)
        if 'healer' in globals():
            st.info("🔄 Attempting self-healing...")
            healer.heal_from_error(main, e)
