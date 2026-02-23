"""
chemnanobew_app.py – Synthesis Optimization Suite (RDKit‑free)
Deploy on Streamlit Cloud with no RDKit dependency.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from PIL import Image
import os
import io
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
# Molecular utilities – RDKit‑free fallback
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
# Tab: Quantum Dots (unchanged, uses DataManager)
# ============================================================================
def display_quantum_dots_tab(uploaded_file):
    st.markdown("<h2 class='sub-header'>CIS/ZnS Quantum Dot Synthesis Optimization</h2>", unsafe_allow_html=True)
    data = uploaded_file and DataManager.load_data(uploaded_file) or DataManager.create_sample_qd_data(50)
    if data is None:
        st.error("No data available")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "🔬 Optimization", "📈 Visualization", "📥 Export"])
    with tab1:
        col1, col2 = st.columns([2,1])
        with col1:
            st.dataframe(data.head(10), use_container_width=True)
            with st.expander("Summary Statistics"):
                st.dataframe(data.describe())
        with col2:
            st.metric("Total Experiments", len(data))
            for t in ['absorption_nm','plqy_percent','pce_percent','soq_au']:
                if t in data:
                    st.metric(f"Best {t}", f"{data[t].max():.1f}")
    with tab2:
        st.markdown("### Optimization Settings")
        target = st.selectbox("Target Property", [c for c in data.columns if c not in ['surfactant','solvent']])
        if st.button("Run Optimization"):
            with st.spinner("Optimizing..."):
                time.sleep(2)
                best = data[target].max() * (1 + np.random.uniform(0.05,0.15))
                st.success(f"Optimal value: {best:.2f}")
    with tab3:
        num_cols = data.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            x = st.selectbox("X axis", num_cols, index=0)
            y = st.selectbox("Y axis", num_cols, index=min(1,len(num_cols)-1))
            fig = px.scatter(data, x=x, y=y, trendline="lowess")
            st.plotly_chart(fig)
    with tab4:
        csv = data.to_csv(index=False)
        st.download_button("Download CSV", csv, "qd_data.csv", "text/csv")

# ============================================================================
# Tab: Porphyrins (unchanged except property prediction now uses MolecularUtils)
# ============================================================================
def display_porphyrins_tab(uploaded_file):
    st.markdown("<h2 class='sub-header'>Porphyrin Synthesis Optimization</h2>", unsafe_allow_html=True)
    data = uploaded_file and DataManager.load_data(uploaded_file) or DataManager.create_sample_porphyrin_data(50)
    if data is None:
        return

    tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "🔬 Synthesis Optimization", "🧪 Property Prediction"])
    with tab1:
        st.dataframe(data.head(10))
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Yield", f"{data['yield_percent'].mean():.1f}%")
        col2.metric("Avg Purity", f"{data['purity_percent'].mean():.1f}%")
        col3.metric("Best SOQ", f"{data['singlet_oxygen_au'].max():.3f}")
    with tab2:
        st.markdown("### Predict Yield")
        temp = st.slider("Temperature (°C)", 20,150,80)
        time_r = st.slider("Reaction time (min)", 30,1440,720)
        cat_conc = st.slider("Catalyst conc (M)", 0.001,0.05,0.01, format="%.3f")
        if st.button("Predict"):
            pred = 45 + (temp-80)*0.1 + (time_r-720)*0.01 + cat_conc*100
            pred = max(10, min(85, pred))
            st.success(f"Predicted yield: {pred:.1f}%")
    with tab3:
        smiles = st.text_input("Enter SMILES", "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=N5)C=C2")
        if st.button("Calculate Properties"):
            props = MolecularUtils.estimate_properties(smiles)
            if props:
                for k,v in props.items():
                    st.metric(k.replace('_',' ').title(), v)
            else:
                st.error("Invalid SMILES")

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
# Tab: Molecular Generator (now using RDKit‑free MolecularUtils)
# ============================================================================
def display_molecular_generator_tab():
    st.markdown("<h2 class='sub-header'>Porphyrin Generator</h2>", unsafe_allow_html=True)
    st.info("RDKit not available – showing SMILES and estimated properties only.")
    n_mols = st.slider("Number of molecules", 5, 20, 10)
    if st.button("Generate"):
        utils = MolecularUtils()
        smiles_list = utils.generate_porphyrin_variants(n_mols)
        for i, smi in enumerate(smiles_list):
            with st.expander(f"Molecule {i+1}"):
                st.code(smi)
                props = utils.estimate_properties(smi)
                if props:
                    for k,v in props.items():
                        st.text(f"{k}: {v}")

# ============================================================================
# Tab: Deepseek AI Assistant (unchanged)
# ============================================================================
class DeepseekChatbot:
    def __init__(self):
        self.history = []
    def get_response(self, msg):
        m = msg.lower()
        if 'quantum' in m or 'qd' in m:
            return "For CIS/ZnS QDs: Cu:In ratio 0.8‑1.2, T 180‑220°C, time 60‑120 min."
        if 'porphyrin' in m:
            return "Lindsey method: 10 mM aldehyde/pyrrole, BF3·OEt2, DDQ oxidation."
        if 'optim' in m:
            return "Use Bayesian optimization with Gaussian processes."
        return "Ask about QDs, porphyrins, or optimization!"

def display_deepseek_chatbox():
    st.markdown("<h2 class='sub-header'>🤖 AI Assistant</h2>", unsafe_allow_html=True)
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = DeepseekChatbot()
        st.session_state.messages = [{"role":"assistant","content":"Hi! How can I help?"}]
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask about synthesis..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        resp = st.session_state.chatbot.get_response(prompt)
        st.session_state.messages.append({"role":"assistant","content":resp})
        with st.chat_message("assistant"):
            st.markdown(resp)

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
        st.info("CHEM‑NANO‑BEW Lab • v2.1 (RDKit‑free)")

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

    st.markdown("<div class='footer'>Powered by CHEMNANOBEW GROUP • RDKit‑free version</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
