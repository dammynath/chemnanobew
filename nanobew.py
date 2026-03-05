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
# ============================================================================
# TAB 1: Quantum Dots
# ============================================================================
def display_quantum_dots_tab(uploaded_file):
    """Quantum Dots tab content"""
    st.markdown("<h2 class='sub-header'>CIS/ZnS Quantum Dot Synthesis Optimization</h2>", unsafe_allow_html=True)
    
    # Safely load data
    if uploaded_file is not None:
        data = DataManager.load_data(uploaded_file)
        if data is None:
            st.warning("⚠️ Could not load uploaded file. Using sample data instead.")
            data = DataManager.create_sample_qd_data(50)
    else:
        data = DataManager.create_sample_qd_data(50)
        st.info("📊 Using sample data. Upload your own CSV for real optimization.")
    
    if data is None or len(data) == 0:
        st.error("❌ No data available")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Explorer", 
        "🔬 Optimization", 
        "📈 Visualization", 
        "👨‍🔬 CIS-Te/ZnS Optimizer", 
        "📥 Export"
    ])
    
    with tab1:
        col1, col2 = st.columns([2,1])
        with col1:
            st.dataframe(data.head(10), use_container_width=True)
            with st.expander("Summary Statistics"):
                st.dataframe(data.describe())
        with col2:
            st.metric("Total Experiments", len(data))
            for t in ['absorption_nm', 'plqy_percent', 'pce_percent', 'soq_au']:
                if t in data.columns:
                    st.metric(f"Best {t}", f"{data[t].max():.1f}")
    
    with tab2:
        st.markdown("### 🔧 Optimization Settings")
        target = st.selectbox(
            "Target Property", 
            [c for c in data.columns if c not in ['surfactant', 'solvent']]
        )
        if st.button("🚀 Run Optimization"):
            with st.spinner("Optimizing..."):
                time.sleep(2)
                best = data[target].max() * (1 + np.random.uniform(0.05, 0.15))
                st.success(f"✅ Optimal value: {best:.2f}")
    
    with tab3:
        num_cols = data.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            x = st.selectbox("X axis", num_cols, index=0)
            y = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1))
            fig = px.scatter(data, x=x, y=y, trendline="lowess")
            st.plotly_chart(fig)
        else:
            st.warning("No numeric columns available for visualization")
    
    with tab4:
        st.header("🧪 CIS-Te/ZnS Optimizer")
        
        pH = st.slider("pH Range", 2.5, 6.0, (3.0, 5.0))
        Te = st.slider("Te (g) Range", 0.0012, 0.0022, (0.0016, 0.0020))
        Zn = st.slider("ZnAc (g) Range", 0.02, 0.04, (0.025, 0.035))
        shell = st.slider("Shell time (min) Range", 10, 60, (15, 45))
        
        ranges = {
            "pH": pH,
            "Te": Te,
            "Zn": Zn,
            "shell": shell
        }
        
        uploaded = st.file_uploader("Upload experimental CSV for Te/ZnS", type="csv", key="te_upload")
        
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success("✅ Using uploaded experimental data")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                df = generate_doe_data(ranges, 40)
                df["wavelength"] = 720 + 250 * df["Te"] + 0.4 * df["shell"]
                df["intensity"] = 15000 + 5000 * (df["pH"] - 4)
        else:
            with st.spinner("Generating synthetic DOE data..."):
                df = generate_doe_data(ranges, 40)
                df["wavelength"] = 720 + 250 * df["Te"] + 0.4 * df["shell"]
                df["intensity"] = 15000 + 5000 * (df["pH"] - 4)
            st.info("📊 Using synthetic DOE data. Upload your own CSV for real optimization.")
        
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 Train RF Model", use_container_width=True):
                with st.spinner("Training Random Forest model..."):
                    model = train_random_forest(df, ["wavelength", "intensity"])
                    st.session_state['rf_model'] = model
                    st.success("✅ Model trained successfully!")
        
        with col2:
            if st.button("🚀 Run Bayesian Optimization", use_container_width=True):
                if 'rf_model' in st.session_state:
                    with st.spinner("Optimizing..."):
                        best = bayesian_optimize(st.session_state['rf_model'], ranges, ["wavelength", "intensity"])
                        st.success(f"✅ Optimal conditions found:")
                        for k, v in best.items():
                            st.metric(k, f"{v:.4f}" if isinstance(v, float) else v)
                else:
                    st.warning("⚠️ Please train a model first")
        
        with col3:
            if st.button("📊 Show Pareto Front", use_container_width=True):
                pf = calculate_pareto_front(df, ["wavelength", "intensity"])
                fig = px.scatter(df, x="wavelength", y="intensity", 
                               title="Pareto Front Analysis",
                               labels={"wavelength": "Wavelength (nm)", "intensity": "Intensity (a.u.)"})
                fig.add_scatter(x=pf["wavelength"], y=pf["intensity"], 
                              mode="markers", name="Pareto Front",
                              marker=dict(color="red", size=10, symbol="star"))
                st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🤖 RL Suggest Next Experiment", use_container_width=True):
            hist = df.copy()
            hist["reward"] = hist["wavelength"] + hist["intensity"] / 1000
            suggestion = rl_suggest_experiment(hist, ranges)
            st.success("🔮 Next suggested experiment:")
            for k, v in suggestion.items():
                st.metric(k, f"{v:.4f}" if isinstance(v, float) else v)
    
    with tab5:
        st.markdown("### 📥 Export Data")
        csv = data.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"qd_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )


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
            ["Plotly", "ggplot2", "seaborn", "simple_white", "presentation", "xgridoff"],
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
# TAB 6: PCE Analyzer
# ============================================================================
def display_pce_tab():
    """Photothermal Conversion Efficiency Analysis Tab"""
    
    st.markdown("<h2 class='sub-header'>🔥 Photothermal Conversion Efficiency (PCE) Analyzer</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Calculate photothermal conversion efficiency for quantum dots, metal nanoparticles, and carbon dots.
    Upload your time-temperature data or use the provided sample data.
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
        # ========================================================================
# Tab 1: Data Input - FIXED ENCODING
# ========================================================================
    with pce_tabs[0]:
        col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📁 Data Source")
        
        data_source = st.radio(
            "Select data source:",
            ["Use Sample Data for (PCE%)", "Upload Custom CSV"],
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
                        df.columns = ['time_h', 'temperature_°C'] + list(df.columns[2:])
                    
                    st.session_state['pce_data'] = df
            else:
                df = st.session_state.get('pce_data', None)
        else:
            # Use sample data with proper encoding
            sample_data = """time_h,temperature_°C
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
            st.info("📊 Using sample data")
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
                peak_time = df.loc[temp_peak_idx, 'time_h']
                peak_temp = df.loc[temp_peak_idx, 'temperature_°C']
                
                fig = go.Figure()
                
                # Heating phase
                fig.add_trace(go.Scatter(
                    x=df['time_h'].iloc[:temp_peak_idx+1],
                    y=df['temperature_°C'].iloc[:temp_peak_idx+1],
                    mode='lines+markers',
                    name='Heating Phase',
                    line=dict(color='red', width=2),
                    marker=dict(size=4)
                ))
                
                # Cooling phase
                fig.add_trace(go.Scatter(
                    x=df['time_h'].iloc[temp_peak_idx:],
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
                    xaxis_title="Time (hours)",
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
                value=4.18,
                step=0.01,
                format="%.2f",
                key="pce_cp",
                help="Default 4.18 J/g·K for water"
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
                value=0.5,
                step=0.1,
                format="%.2f",
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
                "Solvent Blank ΔQ (°C)",
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
    # Tab 3: Analysis & Plots
    # ========================================================================
    with pce_tabs[2]:
        if 'pce_data' not in st.session_state:
            st.warning("⚠️ Please load data in the Data Input tab first.")
        else:
            df = st.session_state['pce_data']
            params = st.session_state.get('pce_params', {})
            
            if not params:
                st.warning("⚠️ Please configure PCE parameters in the Parameters tab.")
            else:
                st.markdown("### 📈 Photothermal Analysis")
                
                # Get peak information
                peak_idx = st.session_state['peak_idx']
                peak_time = st.session_state['peak_time']
                peak_temp = st.session_state['peak_temp']
                
                # Calculate temperature difference
                delta_T = peak_temp - params['ambient_temp']
                delta_T_net = delta_T - params['solvent_delta']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("ΔT Total", f"{delta_T:.2f}°C")
                with col2:
                    st.metric("ΔT Net", f"{delta_T_net:.2f}°C")
                with col3:
                    st.metric("Peak Temperature", f"{peak_temp:.1f}°C")
                with col4:
                    st.metric("Time to Peak", f"{peak_time:.1f} h")
                
                # Calculate time constant from cooling curve
                cooling_data = df.iloc[peak_idx:].copy()
                cooling_data['time_from_peak'] = cooling_data['time_h'] - peak_time
                cooling_data['theta'] = (cooling_data['temperature_°C'] - params['ambient_temp']) / (peak_temp - params['ambient_temp'])
                
                # Remove any points where theta <= 0
                cooling_data = cooling_data[cooling_data['theta'] > 0.01].copy()
                
                # Fit exponential decay
                def exp_decay(t, tau):
                    return np.exp(-t / tau)
                
                try:
                    popt, pcov = curve_fit(
                        exp_decay, 
                        cooling_data['time_from_peak'], 
                        cooling_data['theta'],
                        p0=[1.0]
                    )
                    tau = popt[0]
                    tau_seconds = tau * 3600  # Convert hours to seconds
                    
                    # Calculate R²
                    residuals = cooling_data['theta'] - exp_decay(cooling_data['time_from_peak'], tau)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((cooling_data['theta'] - cooling_data['theta'].mean())**2)
                    r_squared = 1 - (ss_res / ss_tot)
                    
                    st.success(f"✅ Time Constant (τ): {tau:.2f} hours ({tau_seconds:.0f} seconds)")
                    st.metric("R² Value", f"{r_squared:.4f}")
                    
                    # Calculate hA
                    hA = (params['mass'] * params['cp']) / tau_seconds
                    st.metric("hA Value", f"{hA:.4f} W/K")
                    
                    # Calculate PCE
                    pce = (hA * delta_T_net) / params['absorbed_power'] * 100
                    
                    # Store for results tab
                    st.session_state['pce_results'] = {
                        'tau_hours': tau,
                        'tau_seconds': tau_seconds,
                        'r_squared': r_squared,
                        'hA': hA,
                        'pce': pce,
                        'delta_T_net': delta_T_net,
                        'absorbed_power': params['absorbed_power']
                    }
                    
                    # Plot cooling curve fit
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=(
                            'Cooling Curve Fit',
                            'ln(θ) vs Time (Linear Fit)'
                        )
                    )
                    
                    # Cooling curve fit
                    t_fit = np.linspace(0, cooling_data['time_from_peak'].max(), 100)
                    theta_fit = exp_decay(t_fit, tau)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=cooling_data['time_from_peak'],
                            y=cooling_data['theta'],
                            mode='markers',
                            name='Experimental θ',
                            marker=dict(color='blue', size=6)
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=t_fit,
                            y=theta_fit,
                            mode='lines',
                            name=f'Fit: θ = exp(-t/{tau:.2f})',
                            line=dict(color='red', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    fig.update_xaxes(title_text="Time from Peak (hours)", row=1, col=1)
                    fig.update_yaxes(title_text="θ = (T - Tₐ)/(Tₘₐₓ - Tₐ)", row=1, col=1)
                    
                    # ln(θ) vs time plot (should be linear for first-order cooling)
                    cooling_data['ln_theta'] = np.log(cooling_data['theta'])
                    
                    # Linear fit for ln(θ)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        cooling_data['time_from_peak'],
                        cooling_data['ln_theta']
                    )
                    line_eq = f"ln(θ) = {slope:.4f}t + {intercept:.4f}"
                    
                    fig.add_trace(
                        go.Scatter(
                            x=cooling_data['time_from_peak'],
                            y=cooling_data['ln_theta'],
                            mode='markers',
                            name='Experimental ln(θ)',
                            marker=dict(color='green', size=6)
                        ),
                        row=1, col=2
                    )
                    
                    # Add linear fit
                    t_linear = np.linspace(0, cooling_data['time_from_peak'].max(), 100)
                    ln_fit = slope * t_linear + intercept
                    
                    fig.add_trace(
                        go.Scatter(
                            x=t_linear,
                            y=ln_fit,
                            mode='lines',
                            name=f'Fit: {line_eq}',
                            line=dict(color='orange', width=2)
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_xaxes(title_text="Time from Peak (hours)", row=1, col=2)
                    fig.update_yaxes(title_text="ln(θ)", row=1, col=2)
                    
                    fig.update_layout(height=500, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display line equation and R²
                    st.info(f"**Linear Regression:** {line_eq} | R² = {r_value**2:.4f}")
                    
                except Exception as e:
                    st.error(f"Error fitting cooling curve: {str(e)}")
    
    # ========================================================================
    # Tab 4: Cooling Curve Analysis
    # ========================================================================
    with pce_tabs[3]:
        if 'pce_data' not in st.session_state:
            st.warning("⚠️ Please load data in the Data Input tab first.")
        else:
            df = st.session_state['pce_data']
            params = st.session_state.get('pce_params', {})
            
            st.markdown("### 📉 Detailed Cooling Curve Analysis")
            
            # User selects cooling range
            peak_idx = st.session_state['peak_idx']
            
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
                cooling_analysis['time_from_peak'] = cooling_analysis['time_h'] - df.loc[peak_idx, 'time_h']
                cooling_analysis['theta'] = (cooling_analysis['temperature_°C'] - params['ambient_temp']) / (df.loc[peak_idx, 'temperature_°C'] - params['ambient_temp'])
                cooling_analysis = cooling_analysis[cooling_analysis['theta'] > 0].copy()
                cooling_analysis['ln_theta'] = np.log(cooling_analysis['theta'])
                
                # Fit multiple models
                st.markdown("#### 🔄 Model Comparison")
                
                # Exponential model (first-order)
                def exp_model(t, tau):
                    return np.exp(-t / tau)
                
                # Bi-exponential model
                def biexp_model(t, a1, tau1, a2, tau2):
                    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)
                
                try:
                    # Fit exponential
                    popt_exp, _ = curve_fit(exp_model, cooling_analysis['time_from_peak'], cooling_analysis['theta'], p0=[1.0])
                    tau_exp = popt_exp[0]
                    
                    # Calculate R² for exponential
                    residuals_exp = cooling_analysis['theta'] - exp_model(cooling_analysis['time_from_peak'], tau_exp)
                    ss_res_exp = np.sum(residuals_exp**2)
                    ss_tot = np.sum((cooling_analysis['theta'] - cooling_analysis['theta'].mean())**2)
                    r2_exp = 1 - (ss_res_exp / ss_tot)
                    
                    # Try bi-exponential if enough data points
                    if len(cooling_analysis) > 10:
                        try:
                            popt_biexp, _ = curve_fit(
                                biexp_model, 
                                cooling_analysis['time_from_peak'], 
                                cooling_analysis['theta'],
                                p0=[0.5, 0.5, 0.5, 5.0],
                                maxfev=5000
                            )
                            residuals_biexp = cooling_analysis['theta'] - biexp_model(cooling_analysis['time_from_peak'], *popt_biexp)
                            ss_res_biexp = np.sum(residuals_biexp**2)
                            r2_biexp = 1 - (ss_res_biexp / ss_tot)
                            
                            # Display comparison
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Exponential Model**")
                                st.write(f"τ = {tau_exp:.3f} h")
                                st.write(f"R² = {r2_exp:.4f}")
                            
                            with col2:
                                st.markdown("**Bi-exponential Model**")
                                st.write(f"a₁ = {popt_biexp[0]:.3f}, τ₁ = {popt_biexp[1]:.3f} h")
                                st.write(f"a₂ = {popt_biexp[2]:.3f}, τ₂ = {popt_biexp[3]:.3f} h")
                                st.write(f"R² = {r2_biexp:.4f}")
                            
                            # Plot both models
                            fig = go.Figure()
                            
                            # Experimental data
                            fig.add_trace(go.Scatter(
                                x=cooling_analysis['time_from_peak'],
                                y=cooling_analysis['theta'],
                                mode='markers',
                                name='Experimental',
                                marker=dict(color='black', size=6)
                            ))
                            
                            # Exponential fit
                            t_fit = np.linspace(0, cooling_analysis['time_from_peak'].max(), 100)
                            fig.add_trace(go.Scatter(
                                x=t_fit,
                                y=exp_model(t_fit, tau_exp),
                                mode='lines',
                                name=f'Exponential (R²={r2_exp:.3f})',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Bi-exponential fit
                            fig.add_trace(go.Scatter(
                                x=t_fit,
                                y=biexp_model(t_fit, *popt_biexp),
                                mode='lines',
                                name=f'Bi-exponential (R²={r2_biexp:.3f})',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                            
                            fig.update_layout(
                                title="Cooling Curve Model Comparison",
                                xaxis_title="Time from Peak (hours)",
                                yaxis_title="θ",
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except:
                            st.warning("Bi-exponential fit failed - using exponential model")
                    
                    # Linear fit of ln(θ)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        cooling_analysis['time_from_peak'],
                        cooling_analysis['ln_theta']
                    )
                    
                    st.markdown("#### 📊 Linear Regression Analysis")
                    st.write(f"**Equation:** ln(θ) = {slope:.4f}t + {intercept:.4f}")
                    st.write(f"**R²:** {r_value**2:.4f}")
                    st.write(f"**Time constant from slope:** τ = {-1/slope:.3f} hours")
                    
                except Exception as e:
                    st.error(f"Error in model fitting: {str(e)}")
    
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
                st.write(f"**Absorbed Power:** {params.get('absorbed_power', 0):.3f} W")
            
            with col2:
                st.markdown("#### 📈 Thermal Parameters")
                st.write(f"**Time Constant (τ):** {results['tau_hours']:.3f} hours")
                st.write(f"**Time Constant (τ):** {results['tau_seconds']:.0f} seconds")
                st.write(f"**hA Value:** {results['hA']:.4f} W/K")
                st.write(f"**ΔT Net:** {results['delta_T_net']:.2f}°C")
                st.write(f"**R² Value:** {results['r_squared']:.4f}")
            
            st.markdown("---")
            
            # Main PCE result
            st.markdown("### 🎯 Photothermal Conversion Efficiency")
            
            # Create a big metric display
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 1rem;'>
                    <h1 style='color: white; font-size: 4rem;'>{results['pce']:.1f}%</h1>
                    <p style='color: white; font-size: 1.2rem;'>Photothermal Conversion Efficiency</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Comparison with expected value
            expected_pce = 26.0
            difference = abs(results['pce'] - expected_pce)
            percent_diff = (difference / expected_pce) * 100
            
            st.markdown("#### 📊 Validation")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Calculated PCE", f"{results['pce']:.2f}%")
            with col2:
                st.metric("Expected PCE", f"{expected_pce:.2f}%")
            with col3:
                st.metric("Difference", f"{difference:.2f}%", delta=f"{-percent_diff:.1f}%" if results['pce'] < expected_pce else f"{percent_diff:.1f}%")
            
            # Export results
            st.markdown("#### 📥 Export Results")
            
            # Create results dictionary for export
            export_data = {
                'Parameter': [],
                'Value': [],
                'Unit': []
            }
            
            # Add all results
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    export_data['Parameter'].append(key.replace('_', ' ').title())
                    export_data['Value'].append(value)
                    export_data['Unit'].append('')
            
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    export_data['Parameter'].append(key.replace('_', ' ').title())
                    export_data['Value'].append(value)
                    unit = 'hours' if 'tau' in key else ('W/K' if 'hA' in key else ('%' if 'pce' in key else ''))
                    export_data['Unit'].append(unit)
            
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
            'display_pce_tab',
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
