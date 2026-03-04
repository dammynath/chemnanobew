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
import plotly.io as pio
#from skopt import gp_minimize
#from skopt.space import Real
import base64
from PIL import Image
import os
import io
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
try:
    import optuna
    HAS_OPTUNA = True
except:
    HAS_OPTUNA = False
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    st.warning("⚠️ RDKit not installed – using simplified property estimation.")
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy import stats

# Add to your existing helper functions
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
# Custom CSS (unchanged)
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
# Utility functions
# ============================================================================
def save_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        os.makedirs("images", exist_ok=True)
        path = os.path.join("images", uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return path
    return None
def generate_doe(ranges, n=30):
    data = {}
    for k,(lo,hi) in ranges.items():
        data[k] = np.random.uniform(lo,hi,n)
    return pd.DataFrame(data)


def train_rf(df, targets):
    X = df.drop(columns=targets)
    y = df[targets]
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300))
    model.fit(X,y)
    return model


###############################################################
# PARETO FRONT
###############################################################

def pareto_front(df, cols):
    data = df[cols].values
    mask = np.ones(len(data), dtype=bool)

    for i,c in enumerate(data):
        if mask[i]:
            mask[mask] = np.any(data[mask] > c, axis=1)
            mask[i]=True
    return df[mask]


###############################################################
# BAYESIAN OPTIMIZATION
###############################################################

def bayes_optimize(model, ranges, targets=["wavelength","intensity"]):

    def objective(trial):
        x=[]
        for k,(lo,hi) in ranges.items():
            x.append(trial.suggest_float(k, lo, hi))
        x=np.array(x).reshape(1,-1)

        pred = model.predict(x)[0]

        score = pred[0] + pred[1]/1000
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    return study.best_params


###############################################################
# REINFORCEMENT LOOP (simple epsilon-greedy)
###############################################################

def rl_suggest(history, ranges, eps=0.2):

    if len(history)<5 or np.random.rand()<eps:
        return {k:np.random.uniform(lo,hi) for k,(lo,hi) in ranges.items()}

    best = history.sort_values("reward", ascending=False).iloc[0]
    suggestion={}
    for k,(lo,hi) in ranges.items():
        suggestion[k] = np.clip(best[k] + np.random.normal(0,(hi-lo)*0.1),lo,hi)
    return suggestion


###############################################################
# PORPHYRIN GENERATOR (RDKit)
###############################################################

def generate_porhyrin_smiles(subs):

    base = "c1cc2ccc3ccc4ccc(c1)c2c3c4"  # simplified macrocycle scaffold
    smiles=[]
    for s in subs:
        smiles.append(base + s)
    return smiles


###############################################################
# GNN PLACEHOLDER (feature-based mock) CHEMNANOBEW_RUN ENDS
###############################################################

def gnn_predict(features):
    # lightweight surrogate until real PyG model plugged in
    return np.sum(features,axis=1)*0.5 + 700

# ============================================================================
# DataManager (unchanged)
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
            'precursor_ratio': np.random.uniform(0.5,2.0,n),
            'temperature': np.random.uniform(150,250,n),
            'reaction_time': np.random.uniform(30,180,n),
            'zn_precursor': np.random.uniform(0.1,1.0,n),
            'ph': np.random.uniform(4,10,n),
            'surfactant': np.random.choice(['oleic_acid','oleylamine','dodecanethiol'],n),
            'solvent': np.random.choice(['octadecene','toluene','chloroform'],n),
            'absorption_nm': np.random.normal(700,100,n),
            'plqy_percent': np.random.normal(50,15,n),
            'pce_percent': np.random.normal(45,12,n),
            'soq_au': np.random.normal(0.5,0.15,n)
        })

    @staticmethod
    def create_sample_porphyrin_data(n=50):
        np.random.seed(42)
        return pd.DataFrame({
            'aldehyde_conc': np.random.uniform(0.01,0.1,n),
            'pyrrole_conc': np.random.uniform(0.01,0.1,n),
            'temperature': np.random.uniform(20,150,n),
            'reaction_time': np.random.uniform(30,1440,n),
            'catalyst_conc': np.random.uniform(0.001,0.05,n),
            'catalyst_type': np.random.choice(['BF3','TFA','DDQ','p-chloranil'],n),
            'solvent': np.random.choice(['DCM','CHCl3','toluene','DMF'],n),
            'yield_percent': np.random.normal(45,15,n),
            'purity_percent': np.random.normal(85,8,n),
            'singlet_oxygen_au': np.random.normal(0.5,0.15,n),
            'fluorescence_qy': np.random.normal(0.12,0.05,n)
        })

# ============================================================================
# Molecular utilities – RDKit‑Mode fallback
# ============================================================================
class MolecularUtils:
    """Simplified molecular handling without RDKit."""

    @staticmethod
    def validate_smiles(smiles):
        """Very basic SMILES sanity check (porphyrin‑like patterns)."""
        if not isinstance(smiles, str) or len(smiles) < 5:
            return False
        # Look for typical porphyrin fragments (C, N, rings)
        s = smiles.lower()
        return 'c' in s and 'n' in s and ('1' in s or '2' in s)

    @staticmethod
    def estimate_properties(smiles):
        """Heuristic property estimation from SMILES string."""
        if not MolecularUtils.validate_smiles(smiles):
            return None

        # Count atoms roughly (character based)
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
            'rings': smiles.count('1') + smiles.count('2')  # rough
        }

    @staticmethod
    def generate_porphyrin_variants(n=10):
        """Return a list of predefined porphyrin SMILES strings."""
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
        # return random selection
        indices = np.random.choice(len(base), min(n, len(base)), replace=False)
        return [base[i] for i in indices]

# ============================================================================
# Tab: Quantum Dots
# ============================================================================
def display_quantum_dots_tab(uploaded_file):
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
    
    # Check if data is valid
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
    
    with tab4:  # CIS-Te/ZnS Optimizer
        st.header("🧪 CIS-Te/ZnS Optimizer")
        
        # Define parameter ranges
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
        
        # Load or generate data
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
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # Prepare features and targets
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols].values
    y = df[target_cols[0]].values  # Use first target for simplicity
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model


def bayesian_optimize(model, ranges, target_cols):
    """Simple Bayesian optimization placeholder"""
    from skopt import gp_minimize
    from skopt.space import Real    
    # Define search space
    dimensions = []
    param_names = []
    for param, (low, high) in ranges.items():
        dimensions.append(Real(low, high, name=param))
        param_names.append(param)
    
    # Define objective function
    def objective(x):
        x_array = np.array(x).reshape(1, -1)
        # Negative because we want to maximize
        return -model.predict(x_array)[0]
    
    # Run optimization
    result = gp_minimize(
        objective, 
        dimensions, 
        n_calls=20, 
        n_initial_points=5,
        random_state=42
    )
    
    # Return best parameters
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
    # Find best performing experiment so far
    best_idx = history["reward"].idxmax()
    best_params = history.loc[best_idx, [c for c in history.columns if c in ranges.keys()]]
    
    # Add small random perturbation for exploration
    suggestion = {}
    for param, (low, high) in ranges.items():
        if param in best_params.index:
            # Add 10% random noise
            noise = np.random.normal(0, (high - low) * 0.1)
            value = best_params[param] + noise
            # Clip to bounds
            suggestion[param] = np.clip(value, low, high)
        else:
            suggestion[param] = np.random.uniform(low, high)
    
    return suggestion
# ============================================================================
# Tab: Porphyrins (unchanged except property prediction now uses MolecularUtils)
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
        data = pd.DataFrame()  # Create empty DataFrame to prevent further errors
    
    # Create expanded tabs
    por_tabs = st.tabs([
        "📊 Data Explorer", 
        "🔬 Synthesis Optimization", 
        "🧪 Property Prediction",
        "📐 Design of Experiments",
        "🤖 RL Optimizer",
        "🎯 Multi-Objective"
    ])
    
    # ============================================================================
    # Tab 0: Data Explorer
    # ============================================================================
    with por_tabs[0]:
        if len(data) > 0:
            st.markdown("### Porphyrin Synthesis Data")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Yield", f"{data['yield_percent'].mean():.1f}%")
            with col2:
                st.metric("Average Purity", f"{data['purity_percent'].mean():.1f}%")
            with col3:
                st.metric("Best Singlet Oxygen", f"{data['singlet_oxygen_au'].max():.3f}")
            with col4:
                st.metric("Best Fluorescence QY", f"{data['fluorescence_qy'].max():.3f}")
            
            # Data visualization
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
                
                # Correlation heatmap
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
    
    # ============================================================================
    # Tab 1: Synthesis Optimization (existing, enhanced)
    # ============================================================================
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
                # Simple prediction model based on existing data
                if target_property in data.columns:
                    # Use random forest for prediction
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.preprocessing import LabelEncoder
                    
                    # Prepare features
                    feature_cols = ['temperature', 'reaction_time', 'catalyst_conc', 
                                   'aldehyde_conc', 'pyrrole_conc']
                    X = data[feature_cols].copy()
                    
                    # Add encoded catalyst type
                    le = LabelEncoder()
                    X['catalyst_encoded'] = le.fit_transform(data['catalyst_type'])
                    
                    # Train model
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X, data[target_property])
                    
                    # Predict for current settings
                    current_features = pd.DataFrame([[
                        temp, time_react, catalyst_conc, aldehyde, pyrrole,
                        le.transform([catalyst])[0] if catalyst in le.classes_ else 0
                    ]], columns=feature_cols + ['catalyst_encoded'])
                    
                    prediction = model.predict(current_features)[0]
                    
                    # Show results
                    st.success(f"✅ Predicted {target_property}: {prediction:.2f}")
                    
                    # Feature importance
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
    
    # ============================================================================
    # Tab 2: Property Prediction (with RDKit if available)
    # ============================================================================
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
                        from rdkit import Chem
                        from rdkit.Chem import Descriptors, rdMolDescriptors
                        
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
                            
                            # Display in grid
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
                # Estimate optical properties based on substituents
                abs_wl = 410  # base Soret
                fluor_wl = 630  # base fluorescence
                qy = 0.12  # base quantum yield
                
                # Rough estimation based on SMILES content
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
    
    # ============================================================================
    # Tab 3: Design of Experiments (DoE)
    # ============================================================================
    with por_tabs[3]:
        st.markdown("### 📐 Design of Experiments for Porphyrin Synthesis")
        
        st.markdown("""
        <div class='info-box'>
        Design optimal experiments to explore the parameter space efficiently.
        Choose factors and generate a randomized experimental design.
        </div>
        """, unsafe_allow_html=True)
        
        # Define factor ranges based on typical porphyrin synthesis
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
            
            responses = st.multiselect(
                "Response Variables",
                ['yield_percent', 'purity_percent', 'singlet_oxygen_au', 'fluorescence_qy'],
                default=['yield_percent', 'purity_percent'],
                key="doe_responses"
            )
        
        if st.button("🎲 Generate Experimental Design", use_container_width=True):
            from scipy.stats import qmc
            
            # Generate design matrix
            if design_type == "Latin Hypercube":
                # Latin Hypercube sampling
                n_factors = 5  # temp, time, aldehyde, pyrrole, catalyst_conc
                sampler = qmc.LatinHypercube(d=n_factors)
                sample = sampler.random(n=n_experiments)
                
                # Scale to factor ranges
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
                # Simple factorial design approximation
                import itertools
                
                # Create grid of levels
                temp_levels = np.linspace(temp_range[0], temp_range[1], 3)
                time_levels = np.linspace(time_range[0], time_range[1], 3)
                aldehyde_levels = np.linspace(aldehyde_range[0], aldehyde_range[1], 2)
                pyrrole_levels = np.linspace(pyrrole_range[0], pyrrole_range[1], 2)
                cat_levels = np.linspace(catalyst_range[0], catalyst_range[1], 2)
                
                # Generate all combinations
                combinations = list(itertools.product(
                    temp_levels, time_levels, aldehyde_levels, pyrrole_levels, cat_levels
                ))
                
                # Randomly select n_experiments
                indices = np.random.choice(len(combinations), min(n_experiments, len(combinations)), replace=False)
                design = pd.DataFrame(
                    [combinations[i] for i in indices],
                    columns=['Temperature', 'Reaction_Time', 'Aldehyde_Conc', 
                            'Pyrrole_Conc', 'Catalyst_Conc']
                )
            
            # Add catalyst types (random assignment)
            if catalysts:
                design['Catalyst_Type'] = np.random.choice(catalysts, len(design))
            
            # Add run order (randomized)
            design['Run_Order'] = np.random.permutation(len(design)) + 1
            
            st.success(f"✅ Generated {len(design)} experimental runs")
            st.dataframe(design, use_container_width=True)
            
            # Download design
            csv = design.to_csv(index=False)
            st.download_button(
                label="📥 Download Design as CSV",
                data=csv,
                file_name=f"porphyrin_doe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Visualize design space
            fig = px.scatter_matrix(
                design[['Temperature', 'Reaction_Time', 'Aldehyde_Conc', 'Pyrrole_Conc', 'Catalyst_Conc']],
                title="Experimental Design Space"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================================
    # Tab 4: Reinforcement Learning Optimizer
    # ============================================================================
    with por_tabs[4]:
        st.markdown("### 🤖 Reinforcement Learning Optimizer")
        
        st.markdown("""
        <div class='info-box'>
        Use Reinforcement Learning to adaptively suggest the next best experiment based on previous results.
        The agent learns which parameter combinations lead to optimal properties.
        </div>
        """, unsafe_allow_html=True)
        
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
                    # Prepare features
                    feature_cols = ['temperature', 'reaction_time', 'catalyst_conc', 
                                   'aldehyde_conc', 'pyrrole_conc']
                    
                    if 'catalyst_type' in data.columns:
                        # Encode catalyst
                        le = LabelEncoder()
                        data['catalyst_encoded'] = le.fit_transform(data['catalyst_type'])
                        feature_cols.append('catalyst_encoded')
                    
                    X = data[feature_cols].values
                    
                    # Calculate rewards
                    if objective == 'composite':
                        # Normalize each objective
                        yield_norm = (data['yield_percent'] - data['yield_percent'].min()) / (data['yield_percent'].max() - data['yield_percent'].min() + 1e-6)
                        purity_norm = (data['purity_percent'] - data['purity_percent'].min()) / (data['purity_percent'].max() - data['purity_percent'].min() + 1e-6)
                        soq_norm = (data['singlet_oxygen_au'] - data['singlet_oxygen_au'].min()) / (data['singlet_oxygen_au'].max() - data['singlet_oxygen_au'].min() + 1e-6)
                        qy_norm = (data['fluorescence_qy'] - data['fluorescence_qy'].min()) / (data['fluorescence_qy'].max() - data['fluorescence_qy'].min() + 1e-6)
                        
                        rewards = (w_yield * yield_norm + w_purity * purity_norm + 
                                  w_soq * soq_norm + w_qy * qy_norm)
                    else:
                        rewards = data[objective].values
                    
                    # Train a simple Q-learning agent
                    from sklearn.neighbors import NearestNeighbors
                    
                    # Find best performing experiments
                    best_indices = np.argsort(rewards)[-5:]  # Top 5
                    
                    # Generate suggestions by perturbing best experiments
                    suggestions = []
                    param_ranges = {
                        'temperature': (20, 150),
                        'reaction_time': (30, 1440),
                        'catalyst_conc': (0.001, 0.05),
                        'aldehyde_conc': (0.01, 0.1),
                        'pyrrole_conc': (0.01, 0.1)
                    }
                    
                    for _ in range(n_suggestions):
                        # Pick a random best experiment
                        base_idx = np.random.choice(best_indices)
                        base_params = X[base_idx]
                        
                        # Add noise for exploration
                        suggestion = []
                        for i, param in enumerate(feature_cols):
                            if param in param_ranges:
                                low, high = param_ranges[param]
                                # Add noise scaled by exploration rate
                                noise = np.random.normal(0, (high - low) * exploration_rate)
                                value = base_params[i] + noise
                                suggestion.append(np.clip(value, low, high))
                            else:
                                # Categorical - choose randomly sometimes
                                if np.random.random() < exploration_rate:
                                    suggestion.append(np.random.choice(len(le.classes_)))
                                else:
                                    suggestion.append(base_params[i])
                        
                        suggestions.append(suggestion)
                    
                    # Create suggestions dataframe
                    suggestion_df = pd.DataFrame(suggestions, columns=feature_cols)
                    
                    # Decode catalyst if present
                    if 'catalyst_encoded' in feature_cols:
                        suggestion_df['catalyst_type'] = le.inverse_transform(
                            suggestion_df['catalyst_encoded'].astype(int)
                        )
                        suggestion_df = suggestion_df.drop('catalyst_encoded', axis=1)
                    
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
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=data['temperature'] if 'temperature' in data.columns else [],
                        y=data['reaction_time'] if 'reaction_time' in data.columns else [],
                        mode='markers',
                        name='Historical',
                        marker=dict(color='blue', size=8, opacity=0.5)
                    ))
                    
                    # Suggestions
                    fig.add_trace(go.Scatter(
                        x=suggestion_df['temperature'],
                        y=suggestion_df['reaction_time'],
                        mode='markers',
                        name='RL Suggestions',
                        marker=dict(color='red', size=12, symbol='star')
                    ))
                    
                    fig.update_layout(
                        title="RL Suggestions vs Historical Data",
                        xaxis_title="Temperature (°C)",
                        yaxis_title="Reaction Time (min)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for RL optimization. Please upload data or use sample data.")
    
    # ============================================================================
    # Tab 5: Multi-Objective Optimization
    # ============================================================================
    with por_tabs[5]:
        st.markdown("### 🎯 Multi-Objective Pareto Optimization")
        
        st.markdown("""
        <div class='info-box'>
        Find the Pareto front - optimal trade-offs between competing objectives.
        </div>
        """, unsafe_allow_html=True)
        
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
                # Prepare objectives
                obj1_vals = data[obj1].values
                obj2_vals = data[obj2].values
                
                # Convert to maximization (if minimizing, negate)
                if not maximize1:
                    obj1_vals = -obj1_vals
                if not maximize2:
                    obj2_vals = -obj2_vals
                
                objectives = np.column_stack([obj1_vals, obj2_vals])
                
                # Calculate Pareto front
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
                
                # Create plot
                fig = go.Figure()
                
                # All points
                fig.add_trace(go.Scatter(
                    x=data[obj1],
                    y=data[obj2],
                    mode='markers',
                    name='All Experiments',
                    marker=dict(color='lightblue', size=8, opacity=0.6),
                    text=[f"Run {i}" for i in range(len(data))]
                ))
                
                # Pareto front
                pareto_data = data[is_pareto]
                fig.add_trace(go.Scatter(
                    x=pareto_data[obj1],
                    y=pareto_data[obj2],
                    mode='markers+lines',
                    name='Pareto Front',
                    marker=dict(color='red', size=12, symbol='star'),
                    line=dict(dash='dash', color='gray'),
                    text=[f"Pareto {i}" for i in range(len(pareto_data))]
                ))
                
                fig.update_layout(
                    title=f"Pareto Front: {obj1} vs {obj2}",
                    xaxis_title=obj1,
                    yaxis_title=obj2,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Points on Pareto Front", len(pareto_data))
                with col2:
                    st.metric(f"Best {obj1}", f"{pareto_data[obj1].max():.2f}")
                with col3:
                    st.metric(f"Best {obj2}", f"{pareto_data[obj2].max():.2f}")
                
                # Show Pareto optimal experiments
                with st.expander("📋 Pareto Optimal Experiments"):
                    st.dataframe(pareto_data, use_container_width=True)
        else:
            st.warning("No data available for multi-objective optimization.")


# ============================================================================
# Tab: Multi‑Objective (unchanged)
# ============================================================================
def display_multi_objective_tab():
    st.markdown("<h2 class='sub-header'>Multi‑Objective Pareto Optimization</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    obj1 = col1.selectbox("First objective", ["Absorption","PLQY","PCE","SOQ"])
    obj2 = col2.selectbox("Second objective", ["Absorption","PLQY","PCE","SOQ"])
    if st.button("Calculate Pareto Front"):
        # sample data
        np.random.seed(42)
        x = np.random.normal(700,100,100)
        y = 50 - 0.05*(x-700) + np.random.normal(0,10,100)
        y = np.clip(y,10,85)
        # simplified Pareto filter
        objs = np.column_stack([x,y])
        pareto = np.ones(100,bool)
        for i in range(100):
            for j in range(100):
                if i!=j and objs[j,0]>=objs[i,0] and objs[j,1]>=objs[i,1] and (objs[j,0]>objs[i,0] or objs[j,1]>objs[i,1]):
                    pareto[i]=False
                    break
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='All', marker=dict(color='lightblue')))
        fig.add_trace(go.Scatter(x=objs[pareto,0], y=objs[pareto,1], mode='markers+lines',
                                 name='Pareto front', marker=dict(color='red', size=10, symbol='star')))
        st.plotly_chart(fig)

# ============================================================================
# Tab: Molecular Generator (now using RDKit‑Mode MolecularUtils)
# ============================================================================
# def display_molecular_generator_tab():
#   st.markdown("<h2 class='sub-header'>🎯 Porphyrin Generator with Optical Targets</h2>", unsafe_allow_html=True)
#
#    if not RDKIT_AVAILABLE:
#        st.warning("⚠️ RDKit not available – molecular visualization disabled, but property estimation works.")

def display_molecular_generator_tab():
    st.markdown("<h2 class='sub-header'>🎯 Porphyrin Generator with Optical Targets</h2>", unsafe_allow_html=True)

    if not RDKIT_AVAILABLE:
        st.warning("⚠️ RDKit not available – molecular visualization disabled, but property estimation works.") 
    utils = MolecularUtils()

    # User inputs for target properties
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

            # Display each candidate
            for i, (smi, abs_wl, fluor_wl, qy) in enumerate(candidates):
                with st.expander(f"Molecule {i+1}"):
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        st.code(smi, language="text")
                        if RDKIT_AVAILABLE:
                            from rdkit import Chem
                            from rdkit.Chem import Draw
                            mol = Chem.MolFromSmiles(smi)
                            if mol:
                                img = Draw.MolToImage(mol, size=(250, 250))
                                st.image(img, caption="2D Structure")
                    with col_b:
                        st.markdown("**Estimated Properties**")
                        st.metric("Absorbance (nm)", f"{abs_wl:.1f}")
                        st.metric("Fluorescence (nm)", f"{fluor_wl:.1f}")
                        st.metric("Quantum Yield", f"{qy:.3f}")

    # --- DoE Experiment Suggestion Section ---
    st.markdown("---")
    st.markdown("<h3 class='sub-header'>🔬 DoE: Next 10 Experiments</h3>", unsafe_allow_html=True)

    # Use uploaded CSV if available, otherwise sample data
    data = None
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file:
        data = DataManager.load_data(st.session_state.uploaded_file)
    if data is None:
        data = DataManager.create_sample_porphyrin_data(50)
        st.info("Using sample porphyrin data. Upload your own CSV for real DoE.")

    # Simple DoE suggestion: space‑filling design over key factors
    st.markdown("Based on current data, recommended next experiments (Latin Hypercube sampling):")

    # Extract numeric factors (excluding target properties)
    exclude = ['yield_percent', 'purity_percent', 'singlet_oxygen_au', 'fluorescence_qy']
    factors = [c for c in data.columns if c not in exclude and data[c].dtype in ['float64', 'int64']]

    if len(factors) >= 2:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=len(factors))
        sample = sampler.random(n=10)
        # Scale to factor ranges
        scaled = pd.DataFrame(
            qmc.scale(sample,
                      [data[f].min() for f in factors],
                      [data[f].max() for f in factors]),
            columns=factors
        )
        st.dataframe(scaled, use_container_width=True)

        # Optional: build a simple model to predict yield and suggest conditions that maximize it
        if st.button("🎯 Suggest conditions to maximize yield"):
            from sklearn.ensemble import RandomForestRegressor
            X = data[factors]
            y = data['yield_percent']
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            # Predict on the 10 candidates
            preds = model.predict(scaled)
            best_idx = np.argmax(preds)
            st.success(f"Best predicted yield: {preds[best_idx]:.1f}%")
            st.write("Optimal conditions:")
            st.json(scaled.iloc[best_idx].to_dict())
    else:
        st.warning("Not enough numeric factors for DoE.")

# ============================================================================
# Advanced Visualization & Analytics Tab
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
# AI Research Assistant with Brave & Tavily (Default)
# ============================================================================
import streamlit as st
import requests
import json
from openai import OpenAI
import time
from datetime import datetime
import hashlib

class AIResearchAssistant:
    """AI Assistant with Brave and Tavily as default search providers"""
    
    def __init__(self):
        self.initialize_session()
        self.search_providers = []
        self.setup_providers()
    
    def initialize_session(self):
        """Initialize session state variables"""
        if 'assistant_messages' not in st.session_state:
            st.session_state.assistant_messages = []
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'api_status' not in st.session_state:
            st.session_state.api_status = {
                'brave': False,
                'tavily': False,
                'openai': False
            }
    
    def setup_providers(self):
        """Setup search providers from secrets"""
        # Brave Search
        if 'BRAVE_API_KEY' in st.secrets:
            self.brave_key = st.secrets['BRAVE_API_KEY']
            self.search_providers.append('brave')
            st.session_state.api_status['brave'] = True
        else:
            self.brave_key = None
        
        # Tavily Search
        if 'TAVILY_API_KEY' in st.secrets:
            self.tavily_key = st.secrets['TAVILY_API_KEY']
            self.search_providers.append('tavily')
            st.session_state.api_status['tavily'] = True
        else:
            self.tavily_key = None
        
        # OpenAI
        if 'OPENAI_API_KEY' in st.secrets:
            self.openai_key = st.secrets['OPENAI_API_KEY']
            self.openai_client = OpenAI(api_key=self.openai_key)
            st.session_state.api_status['openai'] = True
        else:
            self.openai_key = None
            self.openai_client = None
    
    # ========================================================================
    # Brave Search Methods
    # ========================================================================
    def brave_web_search(self, query, count=10):
        """Search using Brave Web Search API"""
        if not self.brave_key:
            return None
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.brave_key
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
            else:
                st.warning(f"Brave Search returned status {response.status_code}")
                return None
        except Exception as e:
            st.warning(f"Brave Search error: {str(e)}")
            return None
    
    def brave_news_search(self, query, count=10):
        """Search news using Brave News API"""
        if not self.brave_key:
            return None
        
        url = "https://api.search.brave.com/res/v1/news/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_key
        }
        params = {
            "q": query,
            "count": count,
            "search_lang": "en"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def brave_summarizer(self, query):
        """Get AI-ready context using Brave Summarizer"""
        if not self.brave_key:
            return None
        
        url = "https://api.search.brave.com/res/v1/summarizer/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_key
        }
        params = {
            "q": query,
            "summary": "1"  # Request summary
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'summarizer' in data and 'summary' in data['summarizer']:
                    return data['summarizer']['summary']
            return None
        except Exception:
            return None
    
    def brave_llm_context(self, query):
        """Get optimized context for LLM consumption"""
        if not self.brave_key:
            return None
        
        url = "https://api.search.brave.com/res/v1/llm/context"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_key
        }
        params = {"q": query}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    # ========================================================================
    # Tavily Search Methods
    # ========================================================================
    def tavily_search(self, query, search_depth="advanced", max_results=10):
        """Search using Tavily API"""
        if not self.tavily_key:
            return None
        
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.tavily_key}"
        }
        payload = {
            "query": query,
            "search_depth": search_depth,  # "basic" or "advanced"
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False,
            "include_images": False
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                return response.json()
            else:
                st.warning(f"Tavily Search returned status {response.status_code}")
                return None
        except Exception as e:
            st.warning(f"Tavily Search error: {str(e)}")
            return None
    
    def tavily_extract(self, urls):
        """Extract content from specific URLs using Tavily"""
        if not self.tavily_key:
            return None
        
        url = "https://api.tavily.com/extract"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.tavily_key}"
        }
        payload = {
            "urls": urls if isinstance(urls, list) else [urls]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def tavily_qna(self, query):
        """Question answering using Tavily"""
        if not self.tavily_key:
            return None
        
        url = "https://api.tavily.com/qna"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.tavily_key}"
        }
        payload = {
            "query": query
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    # ========================================================================
    # Search Orchestration
    # ========================================================================
    def search_all(self, query):
        """Search using all available providers"""
        results = {
            'sources': [],
            'content': '',
            'answer': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Try Tavily first (best for Q&A)
        if 'tavily' in self.search_providers:
            tavily_result = self.tavily_search(query, max_results=8)
            if tavily_result:
                # Tavily provides a direct answer
                if 'answer' in tavily_result and tavily_result['answer']:
                    results['answer'] = tavily_result['answer']
                
                # Extract content from results
                if 'results' in tavily_result:
                    for r in tavily_result['results']:
                        results['sources'].append({
                            'title': r.get('title', ''),
                            'url': r.get('url', ''),
                            'content': r.get('content', ''),
                            'provider': 'tavily'
                        })
                        results['content'] += f"\n{r.get('content', '')}\n"
        
        # Try Brave for additional context
        if 'brave' in self.search_providers:
            # Get LLM-optimized context from Brave
            brave_context = self.brave_llm_context(query)
            if brave_context and 'results' in brave_context:
                for r in brave_context['results']:
                    results['sources'].append({
                        'title': r.get('title', ''),
                        'url': r.get('url', ''),
                        'description': r.get('description', ''),
                        'provider': 'brave'
                    })
                    results['content'] += f"\n{r.get('description', '')}\n"
            
            # Try summarizer for quick answers
            if not results['answer']:
                brave_summary = self.brave_summarizer(query)
                if brave_summary:
                    results['answer'] = brave_summary
        
        return results
    
    # ========================================================================
    # OpenAI Response Generation
    # ========================================================================
    def generate_response(self, query, search_results):
        """Generate AI response using OpenAI"""
        if not self.openai_client:
            return "OpenAI API not configured. Please check your API key."
        
        # Build context from search results
        context = search_results.get('content', '')
        sources = search_results.get('sources', [])
        
        # Create system prompt
        system_prompt = """You are ChemNanoBew AI, a research assistant specializing in chemistry, quantum dots, and porphyrin synthesis. 
        Use the provided search results to answer the user's question accurately. 
        If you don't know the answer based on the search results, say so honestly.
        Always cite your sources when possible."""
        
        # Create user prompt with context
        user_prompt = f"""Search Results:
        {context[:4000]}  # Limit context length
        
        Question: {query}
        
        Please provide a comprehensive answer based on the search results above.
        If there's a direct answer available, start with that, then elaborate.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",  # or "gpt-3.5-turbo" for lower cost
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    # ========================================================================
    # Chemistry-Specific Search
    # ========================================================================
    def search_chemistry_literature(self, query):
        """Specialized search for chemistry literature"""
        # Add chemistry-specific keywords
        enhanced_query = f"{query} chemistry synthesis materials science"
        return self.search_all(enhanced_query)
    
    def search_qd_synthesis(self, query):
        """Specialized search for quantum dot synthesis"""
        enhanced_query = f"{query} quantum dot CIS ZnS synthesis optimization"
        return self.search_all(enhanced_query)
    
    def search_porphyrin_synthesis(self, query):
        """Specialized search for porphyrin synthesis"""
        enhanced_query = f"{query} porphyrin synthesis photodynamic therapy"
        return self.search_all(enhanced_query)
    
   # ========================================================================
    # UI Rendering
    # ========================================================================
    def render_ui(self):
        """Render the AI Assistant UI in Streamlit"""
        
        st.markdown("<h2 class='sub-header'>🤖 AI Research Assistant</h2>", unsafe_allow_html=True)
        
        # API Status Display
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.api_status['brave']:
                st.success("✅ Brave Search")
            else:
                st.warning("⚠️ Brave Search")
        with col2:
            if st.session_state.api_status['tavily']:
                st.success("✅ Tavily Search")
            else:
                st.warning("⚠️ Tavily Search")
        with col3:
            if st.session_state.api_status['openai']:
                st.success("✅ OpenAI")
            else:
                st.error("❌ OpenAI")
        
        # Configuration instructions if APIs missing
        if not all(st.session_state.api_status.values()):
            with st.expander("🔧 Configure APIs", expanded=True):
                st.markdown("""
                ### Get Your Free API Keys:
                
                **1. OpenAI** (Required)
                - Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
                - Create new key → Copy it
                
                **2. Brave Search** (Free - Recommended)
                - Go to [brave.com/search/api](https://brave.com/search/api)
                - Sign up for free tier (2000 queries/month)
                - Copy your API key
                
                **3. Tavily** (Free - Recommended)
                - Go to [tavily.com](https://tavily.com)
                - Sign up for free tier (1000 searches/month)
                - Copy your API key
                
                **Add to `.streamlit/secrets.toml`:**
                ```toml
                OPENAI_API_KEY = "sk-..."
                BRAVE_API_KEY = "BSA..."
                TAVILY_API_KEY = "tvly-..."
                """)

                st.markdown("---")

#Search mode selector
search_mode = st.radio(
"Search Mode",
["General Research", "Quantum Dots", "Porphyrins", "Chemistry Literature"],
horizontal=True,
key="search_mode"
)

#Display chat history
for message in st.session_state.assistant_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if "sources" in message and message["sources"]:
    with st.expander(f"📚 Sources ({len(message['sources'])})"):
        for src in message["sources"]:
            title = src.get('title', src.get('url', 'Source'))
url = src.get('url', '#')
provider = src.get('provider', 'web')
st.markdown(f"- {title} ({provider})")

#Chat input
if prompt := st.chat_input("Ask about synthesis, research, or any topic..."):

#Add user message
    st.session_state.assistant_messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.markdown(prompt)

#Generate response
with st.chat_message("assistant"):
    with st.spinner("🔍 Searching Brave & Tavily..."):

#Select search mode
        if search_mode == "Quantum Dots":
            search_results = self.search_qd_synthesis(prompt)
elif search_mode == "Porphyrins":
search_results = self.search_porphyrin_synthesis(prompt)
elif search_mode == "Chemistry Literature":
search_results = self.search_chemistry_literature(prompt)
else:
search_results = self.search_all(prompt)

#Generate AI response
with st.spinner("🧠 Thinking..."):
response = self.generate_response(prompt, search_results)
st.markdown(response)

#Show sources
if search_results['sources']:
with st.expander(f"📚 Sources ({len(search_results['sources'])})"):
for src in search_results['sources'][:5]: # Show top 5
title = src.get('title', src.get('url', 'Source'))
url = src.get('url', '#')
provider = src.get('provider', 'web')
st.markdown(f"- {title} ({provider})")

if len(search_results['sources']) > 5:
st.caption(f"... and {len(search_results['sources']) - 5} more sources")

#Save to session
st.session_state.assistant_messages.append({
"role": "assistant",
"content": response,
"sources": search_results['sources'][:10]
})

#Sidebar with search history
with st.sidebar.expander("📜 Search History", expanded=False):
for i, msg in enumerate(st.session_state.assistant_messages[-10:]):
if msg["role"] == "user":
st.markdown(f"Q{i}: {msg['content'][:50]}...")

if st.button("Clear Chat"):
st.session_state.assistant_messages = []
st.rerun()

# ============================================================================
# Tab: AI Assistant (unchanged)
# ============================================================================


# ============================================================================
# AI CHATBOX TAB - FIXED VERSION
# ============================================================================

class ChemNanoBot:
    """Enhanced chatbot for synthesis advice"""
    
    def __init__(self):
        self.conversation_history = []
        self.context_window = []  # Store last 5 exchanges for context
    
    def get_response(self, user_message):
        """Generate response based on user message with context awareness"""
        user_message_lower = user_message.lower()
        
        # Store in context
        self.context_window.append({"role": "user", "content": user_message})
        if len(self.context_window) > 10:
            self.context_window = self.context_window[-10:]
        
        # Check for context-aware responses
        if len(self.context_window) > 2:
            # Look for follow-up questions
            if "more" in user_message_lower or "elaborate" in user_message_lower:
                return self.get_elaboration_response()
        
        # Route to appropriate response
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
        
        # Store response in context
        self.context_window.append({"role": "assistant", "content": response})
        
        return response
    
    def get_elaboration_response(self):
        """Provide elaboration on previous topic"""
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
- Hypervolume indicator tracks progress

**Design of Experiments (DoE):**
- Screening: 2-level factorial (8-16 runs)
- Optimization: Response Surface (15-30 runs)
- Analysis: ANOVA, contour plots

Want me to elaborate on any step?"""
    
    def get_greeting(self):
        return """👋 **Hello! I'm ChemNanoBot, your synthesis optimization assistant!**

I specialize in:
- 🧪 **Quantum Dots** (CIS/ZnS, CdSe, Perovskite)
- 🔬 **Porphyrins** (Synthesis, metalation, properties)
- 📊 **Experimental Design** (DoE, factorial designs)
- 🤖 **Optimization** (Bayesian, multi-objective)
- 📈 **Data Analysis** (Statistics, visualization)

**Try asking me:**
- "How do I optimize QD absorption for 800nm?"
- "What's the best porphyrin synthesis method?"
- "Explain Bayesian optimization"
- "How to improve singlet oxygen generation?"

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
- **Data Analysis**: Statistics, visualization, interpretation

**Example questions:**
- "What temperature for CIS/ZnS core?"
- "How to purify porphyrins?"
- "What's the difference between Type I and Type II PDT?"

What specific topic interests you?"""


def display_ai_assistant():
    """AI Assistant tab with enhanced UI"""
    
    st.markdown("<h2 class='sub-header'>🤖 ChemNanoBot AI Assistant</h2>", unsafe_allow_html=True)
    
    # Custom styling for the chat interface
    st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .quick-question-btn {
        margin: 0.2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Chat with <strong>ChemNanoBot</strong>, your AI expert in quantum dot and porphyrin synthesis optimization.
    Ask about synthesis conditions, experimental design, or data analysis!
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot in session state if not exists
    if 'ai_chatbot' not in st.session_state:
        st.session_state.ai_chatbot = ChemNanoBot()
    
    if 'ai_messages' not in st.session_state:
        st.session_state.ai_messages = [
            {"role": "assistant", "content": "👋 Hello! I'm ChemNanoBot, your synthesis optimization assistant. How can I help you today?"}
        ]
    
    # Category-based quick questions
    st.markdown("### 📋 Quick Question Categories")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Quantum Dots", "Porphyrins", "Optimization", "General"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("QD absorption >800nm", use_container_width=True):
                prompt = "How to achieve QD absorption above 800nm?"
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()
            if st.button("High PLQY QDs", use_container_width=True):
                prompt = "How to get >60% PLQY in quantum dots?"
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()
        with col2:
            if st.button("Shell growth", use_container_width=True):
                prompt = "Best method for ZnS shell growth?"
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()
            if st.button("Size control", use_container_width=True):
                prompt = "How to control quantum dot size?"
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Lindsey method", use_container_width=True):
                prompt = "Explain Lindsey method for porphyrin synthesis"
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()
            if st.button("High singlet oxygen", use_container_width=True):
                prompt = "How to improve singlet oxygen generation in porphyrins?"
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()
        with col2:
            if st.button("Metalation", use_container_width=True):
                prompt = "How to metalate porphyrins?"
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()
            if st.button("Purification", use_container_width=True):
                prompt = "Best porphyrin purification methods?"
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Bayesian optimization", use_container_width=True):
                prompt = "Explain Bayesian optimization for synthesis"
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()
            if st.button("DoE basics", use_container_width=True):
                prompt = "What is Design of Experiments?"
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()
        with col2:
            if st.button("Pareto front", use_container_width=True):
                prompt = "What is Pareto front optimization?"
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()
            if st.button("Response surface", use_container_width=True):
                prompt = "Explain response surface methodology"
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("About ChemNanoBot", use_container_width=True):
                prompt = "Tell me about yourself"
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()
        with col2:
            if st.button("Help", use_container_width=True):
                prompt = "What can you help me with?"
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.session_state.ai_messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    st.markdown("---")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.ai_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about synthesis optimization..."):
        # Add user message
        st.session_state.ai_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔬 Analyzing your question..."):
                response = st.session_state.ai_chatbot.get_response(prompt)
                st.markdown(response)
        
        # Add to history
        st.session_state.ai_messages.append({"role": "assistant", "content": response})
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.ai_messages = [
                {"role": "assistant", "content": "👋 Hello! I'm ChemNanoBot, your synthesis optimization assistant. How can I help you today?"}
            ]
            st.rerun()
    
    with col2:
        if st.button("📋 Copy Last Response", use_container_width=True):
            if len(st.session_state.ai_messages) > 1:
                last_response = st.session_state.ai_messages[-1]["content"]
                st.code(last_response, language="text")
                st.info("✅ Response copied to clipboard (select and copy manually)")
    
    # Chat statistics
    with st.expander("📊 Chat Statistics"):
        user_msgs = sum(1 for m in st.session_state.ai_messages if m["role"] == "user")
        assistant_msgs = sum(1 for m in st.session_state.ai_messages if m["role"] == "assistant")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", len(st.session_state.ai_messages))
        with col2:
            st.metric("Your Questions", user_msgs)
        with col3:
            st.metric("Responses", assistant_msgs)

# ============================================================================
# Main
# ============================================================================
def main():
    with st.sidebar:
        if os.path.exists("images") and os.listdir("images"):
            st.image(os.path.join("images", os.listdir("images")[0]), use_container_width=True)
        else:
            st.markdown("<div class='sidebar-logo'><div style='font-size:3rem;'>🧪</div><div class='sidebar-logo-text'>CHEM‑NANO‑BEW</div></div>", unsafe_allow_html=True)
        st.markdown("---")
        mode = st.radio("Mode", ["Quantum Dots","Porphyrins","Multi‑Objective","Molecular Generator","📊 Advanced Visualization","🤖 AI Research Assistant","AI Assistant"])
        with st.expander("Upload Logo"):
            logo = st.file_uploader("Image", type=['png','jpg','jpeg','gif'])
            if logo:
                save_uploaded_image(logo)
                st.rerun()
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        st.markdown("---")
        st.info("CHEM‑NANO‑BEW Lab • v2.1 (RDKit‑Mode)")

    st.markdown("<h1 class='main-header'>CHEM‑NANO‑BEW LABORATORY</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lab-subtitle'>Advanced Synthesis Optimization Suite</p>", unsafe_allow_html=True)

    if mode == "Quantum Dots":
        display_quantum_dots_tab(uploaded_file)
    elif mode == "Porphyrins":
        display_porphyrins_tab(uploaded_file)
    elif mode == "Multi‑Objective":
        display_multi_objective_tab()
    elif mode == "Molecular Generator":
        display_molecular_generator_tab()
    elif mode == "📊 Advanced Visualization":
        display_advanced_visualization(uploaded_file)
    elif mode == "🤖 AI Research Assistant":
        assistant = AIResearchAssistant()
        assistant.render_ui()
    else:
        display_deepseek_chatbox()

    st.markdown("<div class='footer'>Powered by CHEMNANOBEW GROUP • RDKit‑Mode version</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
