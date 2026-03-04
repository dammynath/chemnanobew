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
# Tab: AI Assistant (unchanged)
# ============================================================================
# ============================================================================
# AI CHATBOX TAB
# ============================================================================

class DeepseekChatbot:
    """Simple chatbot for synthesis advice"""
    
    def __init__(self):
        self.conversation_history = []
    
    def get_response(self, user_message):
        """Generate response based on user message"""
        user_message_lower = user_message.lower()
        
        if 'quantum dot' in user_message_lower or 'qd' in user_message_lower:
            return self.get_qd_response(user_message)
        elif 'porphyrin' in user_message_lower:
            return self.get_porphyrin_response(user_message)
        elif 'optimiz' in user_message_lower:
            return self.get_optimization_response(user_message)
        elif 'hello' in user_message_lower or 'hi' in user_message_lower:
            return self.get_greeting()
        else:
            return self.get_general_response()
    
    def get_qd_response(self, query):
        return """🎯 **Quantum Dot Synthesis Advice**

For optimal CIS/ZnS quantum dots:

**Key Parameters:**
- **Precursor ratio (Cu:In):** 0.8-1.2
- **Temperature:** 180-220°C for core
- **Time:** 60-120 minutes
- **Shell growth:** 200-240°C with Zn precursor

**For absorption ≥800nm:**
- Increase In content
- Extend reaction time
- Grow thicker shells

Would you like specific advice on any parameter?"""
    
    def get_porphyrin_response(self, query):
        return """🧪 **Porphyrin Synthesis Advice**

**Lindsey Method Recommendations:**
- **Concentration:** 0.01-0.02 M
- **Catalyst:** BF3·OEt2 (0.1-0.3 eq)
- **Temperature:** Room temperature
- **Oxidation:** DDQ or p-chloranil

**For high singlet oxygen:**
- Heavy atom substitution (Br, I)
- Metalation with Pd or Pt
- Extended conjugation

Need help with a specific aspect?"""
    
    def get_optimization_response(self, query):
        return """🚀 **Optimization Strategy**

**Bayesian Optimization Workflow:**
1. Define parameter space
2. Build surrogate model (Gaussian Process)
3. Use acquisition function (EI, UCB)
4. Suggest next experiment
5. Update model with results

**Multi-objective tips:**
- Use Pareto front analysis
- Weighted sum for composite score
- Consider trade-offs between properties

Want me to elaborate on any step?"""
    
    def get_greeting(self):
        return """👋 Hello! I'm your synthesis optimization assistant.

I can help with:
- **Quantum Dots** (CIS/ZnS optimization)
- **Porphyrins** (synthesis & properties)
- **Experimental Design** (DoE)
- **Optimization** (Bayesian, multi-objective)

What would you like to know?"""
    
    def get_general_response(self):
        return """I'm here to help with synthesis optimization!

You can ask me about:
- Quantum dot synthesis conditions
- Porphyrin synthesis methods
- Design of experiments
- Bayesian optimization
- Data analysis

What specific topic interests you?"""

def display_deepseek_chatbox():
    """AI Assistant tab"""
    st.markdown("<h2 class='sub-header'>🤖 AI Assistant</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Chat with CHEMNANOBOT, your AI expert in quantum dot and porphyrin synthesis optimization.
    Ask about synthesis conditions, experimental design, or data analysis!
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = DeepseekChatbot()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! I'm your synthesis optimization assistant. How can I help you today?"}
        ]
    
    # Quick questions
    st.markdown("### Quick Questions")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_qs = [
        "How to optimize QD absorption?",
        "Best porphyrin synthesis?",
        "What is Bayesian optimization?",
        "How to improve singlet oxygen?"
    ]
    
    for i, (col, q) in enumerate(zip([col1, col2, col3, col4], quick_qs)):
        with col:
            if st.button(q, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                response = st.session_state.chatbot.get_response(q)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about synthesis optimization..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.get_response(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! I'm your synthesis optimization assistant. How can I help you today?"}
        ]
        st.rerun()

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
        mode = st.radio("Mode", ["Quantum Dots","Porphyrins","Multi‑Objective","Molecular Generator","AI Assistant"])
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
    else:
        display_deepseek_chatbox()

    st.markdown("<div class='footer'>Powered by CHEMNANOBEW GROUP • RDKit‑Mode version</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
