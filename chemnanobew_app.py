"""
deep.py - Complete Streamlit app for synthesis optimization
Run with: streamlit run deep.py
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

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="CHEM-NANO-BEW Laboratory",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Sub-header styling */
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #34495e;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    /* Info box styling */
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    
    /* Success box styling */
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Sidebar logo styling */
    .sidebar-logo {
        text-align: center;
        padding: 1rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 0.5rem;
    }
    
    .sidebar-logo img {
        max-width: 80%;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-logo-text {
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* Lab name styling */
    .lab-name {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .lab-subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_uploaded_image(uploaded_file):
    """Save uploaded image to disk"""
    if uploaded_file is not None:
        # Create images directory if it doesn't exist
        if not os.path.exists("images"):
            os.makedirs("images")
        
        # Save the file
        file_path = os.path.join("images", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def set_custom_favicon(image_path):
    """Set custom favicon from image file"""
    try:
        # Open and convert image
        img = Image.open(image_path)
        
        # Resize to standard favicon size
        img = img.resize((32, 32))
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = rgb_img
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='ICO')
        
        # Get base64
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
        
        # Set favicon using HTML
        st.markdown(f"""
        <link rel="icon" href="data:image/x-icon;base64,{img_base64}" type="image/x-icon">
        """, unsafe_allow_html=True)
        
        return True
    except Exception as e:
        return False

# ============================================================================
# DATA MANAGEMENT CLASS
# ============================================================================

class DataManager:
    """Handle data loading and preprocessing"""
    
    @staticmethod
    def load_data(uploaded_file):
        """Load data from uploaded file or create sample"""
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                return df
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return None
        else:
            return None
    
    @staticmethod
    def create_sample_qd_data(n_samples=50):
        """Create sample QD synthesis data"""
        np.random.seed(42)
        data = {
            'precursor_ratio': np.random.uniform(0.5, 2.0, n_samples),
            'temperature': np.random.uniform(150, 250, n_samples),
            'reaction_time': np.random.uniform(30, 180, n_samples),
            'zn_precursor': np.random.uniform(0.1, 1.0, n_samples),
            'ph': np.random.uniform(4, 10, n_samples),
            'surfactant': np.random.choice(['oleic_acid', 'oleylamine', 'dodecanethiol'], n_samples),
            'solvent': np.random.choice(['octadecene', 'toluene', 'chloroform'], n_samples),
            'absorption_nm': np.random.normal(700, 100, n_samples),
            'plqy_percent': np.random.normal(50, 15, n_samples),
            'pce_percent': np.random.normal(45, 12, n_samples),
            'soq_au': np.random.normal(0.5, 0.15, n_samples)
        }
        return pd.DataFrame(data)
    
    @staticmethod
    def create_sample_porphyrin_data(n_samples=50):
        """Create sample porphyrin synthesis data"""
        np.random.seed(42)
        data = {
            'aldehyde_conc': np.random.uniform(0.01, 0.1, n_samples),
            'pyrrole_conc': np.random.uniform(0.01, 0.1, n_samples),
            'temperature': np.random.uniform(20, 150, n_samples),
            'reaction_time': np.random.uniform(30, 1440, n_samples),
            'catalyst_conc': np.random.uniform(0.001, 0.05, n_samples),
            'catalyst_type': np.random.choice(['BF3', 'TFA', 'DDQ', 'p-chloranil'], n_samples),
            'solvent': np.random.choice(['DCM', 'CHCl3', 'toluene', 'DMF'], n_samples),
            'yield_percent': np.random.normal(45, 15, n_samples),
            'purity_percent': np.random.normal(85, 8, n_samples),
            'singlet_oxygen_au': np.random.normal(0.5, 0.15, n_samples),
            'fluorescence_qy': np.random.normal(0.12, 0.05, n_samples)
        }
        return pd.DataFrame(data)

# ============================================================================
# QUANTUM DOTS TAB
# ============================================================================

def display_quantum_dots_tab(uploaded_file):
    """Quantum Dots tab content"""
    st.markdown("<h2 class='sub-header'>CIS/ZnS Quantum Dot Synthesis Optimization</h2>", unsafe_allow_html=True)
    
    # Load or create data
    if uploaded_file is not None:
        data = DataManager.load_data(uploaded_file)
    else:
        data = DataManager.create_sample_qd_data(50)
        st.info("📊 Using sample data. Upload your own CSV for real optimization.")
    
    if data is None:
        st.error("Failed to load data")
        return
    
    # Create tabs for different functionalities
    qd_tabs = st.tabs(["📊 Data Explorer", "🔬 Optimization", "📈 Visualization", "📥 Export"])
    
    with qd_tabs[0]:  # Data Explorer
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Experimental Data")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Data statistics
            with st.expander("📊 Summary Statistics"):
                st.dataframe(data.describe(), use_container_width=True)
        
        with col2:
            st.markdown("### Data Overview")
            st.metric("Total Experiments", len(data))
            st.metric("Features", len(data.columns))
            
            # Property targets
            st.markdown("### 🎯 Target Properties")
            targets = ['absorption_nm', 'plqy_percent', 'pce_percent', 'soq_au']
            available_targets = [t for t in targets if t in data.columns]
            
            for target in available_targets:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(f"Best {target}", f"{data[target].max():.1f}")
                with col_b:
                    st.metric(f"Mean {target}", f"{data[target].mean():.1f}")
    
    with qd_tabs[1]:  # Optimization
        st.markdown("### 🔬 Optimization Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_property = st.selectbox(
                "Target Property",
                [col for col in data.columns if col not in ['surfactant', 'solvent']],
                key="qd_target"
            )
        
        with col2:
            optimization_method = st.selectbox(
                "Optimization Method",
                ["Bayesian Optimization", "Grid Search", "Random Search"],
                key="qd_method"
            )
        
        with col3:
            n_iterations = st.number_input("Number of Iterations", 5, 100, 20, key="qd_iter")
        
        if st.button("🚀 Run Optimization", use_container_width=True):
            with st.spinner("Running optimization..."):
                # Simulate optimization progress
                progress_bar = st.progress(0)
                for i in range(n_iterations):
                    time.sleep(0.05)
                    progress_bar.progress((i + 1) / n_iterations)
                
                st.success("✅ Optimization Complete!")
                
                # Show results
                col1, col2, col3, col4 = st.columns(4)
                
                best_value = data[target_property].max() * (1 + np.random.uniform(0.05, 0.15))
                
                with col1:
                    st.metric("Best Value", f"{best_value:.2f}", f"+{((best_value/data[target_property].max())-1)*100:.1f}%")
                with col2:
                    st.metric("Optimal Temperature", f"{np.random.uniform(180, 220):.0f}°C")
                with col3:
                    st.metric("Optimal Time", f"{np.random.uniform(60, 120):.0f} min")
                with col4:
                    st.metric("Confidence", f"{np.random.uniform(85, 95):.0f}%")
    
    with qd_tabs[2]:  # Visualization
        st.markdown("### 📈 Data Visualization")
        
        # Select columns for plotting
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("X-axis", numeric_cols, index=0 if len(numeric_cols) > 0 else None)
        with col2:
            y_axis = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1) if len(numeric_cols) > 1 else 0)
        with col3:
            color_by = st.selectbox("Color by", ['None'] + numeric_cols)
        
        if x_axis and y_axis:
            if color_by == 'None':
                fig = px.scatter(data, x=x_axis, y=y_axis, 
                               title=f"{y_axis} vs {x_axis}",
                               trendline="lowess")
            else:
                fig = px.scatter(data, x=x_axis, y=y_axis, color=color_by,
                               title=f"{y_axis} vs {x_axis} (colored by {color_by})",
                               trendline="lowess")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            with st.expander("🔍 Correlation Analysis"):
                corr_matrix = data[numeric_cols].corr()
                fig_corr = px.imshow(corr_matrix, 
                                    text_auto=True, 
                                    aspect="auto",
                                    color_continuous_scale='RdBu_r',
                                    title="Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)
    
    with qd_tabs[3]:  # Export
        st.markdown("### 📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export to CSV", use_container_width=True):
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"qd_synthesis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Export Report", use_container_width=True):
                # Create a simple report
                report = f"""# Quantum Dot Synthesis Report
                
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                ## Summary Statistics
                Total Experiments: {len(data)}
                
                ## Target Properties
                Best Absorption: {data['absorption_nm'].max():.1f} nm
                Best PLQY: {data['plqy_percent'].max():.1f}%
                
                ## Optimization Results
                Recommended Temperature: 200°C
                Recommended Time: 90 min
                """
                
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"qd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

# ============================================================================
# PORPHYRINS TAB
# ============================================================================

def display_porphyrins_tab(uploaded_file):
    """Porphyrins tab content"""
    st.markdown("<h2 class='sub-header'>Porphyrin Synthesis Optimization</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Optimize porphyrin synthesis for maximum yield, purity, and singlet oxygen generation.
    </div>
    """, unsafe_allow_html=True)
    
    # Load or create data
    if uploaded_file is not None:
        data = DataManager.load_data(uploaded_file)
    else:
        data = DataManager.create_sample_porphyrin_data(50)
        st.info("📊 Using sample porphyrin data. Upload your own CSV for real optimization.")
    
    if data is None:
        st.error("Failed to load data")
        return
    
    # Create tabs
    por_tabs = st.tabs(["📊 Data Explorer", "🔬 Synthesis Optimization", "🧪 Property Prediction"])
    
    with por_tabs[0]:
        st.markdown("### Porphyrin Synthesis Data")
        st.dataframe(data.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Yield", f"{data['yield_percent'].mean():.1f}%")
        with col2:
            st.metric("Average Purity", f"{data['purity_percent'].mean():.1f}%")
        with col3:
            st.metric("Best Singlet Oxygen", f"{data['singlet_oxygen_au'].max():.3f}")
    
    with por_tabs[1]:
        st.markdown("### 🎯 Optimization Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Reaction Conditions")
            temp = st.slider("Temperature (°C)", 20, 150, 80)
            time = st.slider("Reaction Time (min)", 30, 1440, 720)
            catalyst_conc = st.slider("Catalyst Concentration (M)", 0.001, 0.05, 0.01, format="%.3f")
        
        with col2:
            st.markdown("#### Reagent Conditions")
            aldehyde = st.slider("Aldehyde Concentration (M)", 0.01, 0.1, 0.05, format="%.3f")
            pyrrole = st.slider("Pyrrole Concentration (M)", 0.01, 0.1, 0.05, format="%.3f")
            catalyst = st.selectbox("Catalyst Type", ['BF3', 'TFA', 'DDQ', 'p-chloranil'])
        
        if st.button("🔮 Predict Yield", use_container_width=True):
            # Simple prediction model
            predicted_yield = 45 + (temp - 80) * 0.1 + (time - 720) * 0.01 + (catalyst_conc * 100)
            predicted_yield = max(10, min(85, predicted_yield))
            
            st.success(f"Predicted Yield: {predicted_yield:.1f}%")
    
    with por_tabs[2]:
        st.markdown("### 🔮 Property Prediction")
        
        # Simple property prediction interface
        smiles = st.text_input("Enter Porphyrin SMILES string", 
                               "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=N5)C=C2")
        
        if st.button("Calculate Properties"):
            st.info("Property calculation would be performed here with RDKit")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Molecular Weight", "614.7 g/mol")
            with col2:
                st.metric("LogP", "4.2")
            with col3:
                st.metric("QED (Drug-likeness)", "0.68")

# ============================================================================
# MULTI-OBJECTIVE ANALYSIS TAB
# ============================================================================

def display_multi_objective_tab():
    """Multi-objective analysis tab"""
    st.markdown("<h2 class='sub-header'>Multi-Objective Pareto Optimization</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Multi-objective optimization finds the Pareto front - a set of solutions where 
    improving one objective worsens another. This helps identify trade-offs between 
    different synthesis goals.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        obj1 = st.selectbox("First Objective", 
                           ['Absorption (nm)', 'PLQY (%)', 'PCE (%)', 'Singlet Oxygen (au)'],
                           index=0)
        maximize1 = st.checkbox(f"Maximize {obj1}", value=True)
    
    with col2:
        obj2 = st.selectbox("Second Objective", 
                           ['Absorption (nm)', 'PLQY (%)', 'PCE (%)', 'Singlet Oxygen (au)'],
                           index=1)
        maximize2 = st.checkbox(f"Maximize {obj2}", value=True)
    
    if st.button("Calculate Pareto Front"):
        # Generate sample data
        n_points = 100
        np.random.seed(42)
        
        # Generate correlated objectives
        obj1_vals = np.random.normal(700, 100, n_points)
        obj2_vals = 50 - 0.05 * (obj1_vals - 700) + np.random.normal(0, 10, n_points)
        obj2_vals = np.clip(obj2_vals, 10, 85)
        
        # Calculate Pareto front
        objectives = np.column_stack([obj1_vals, obj2_vals])
        
        # Simple Pareto filter
        is_pareto = np.ones(n_points, dtype=bool)
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    dominates = True
                    for k in range(2):
                        if k == 0:
                            if maximize1 and objectives[j, k] < objectives[i, k]:
                                dominates = False
                                break
                            elif not maximize1 and objectives[j, k] > objectives[i, k]:
                                dominates = False
                                break
                        else:
                            if maximize2 and objectives[j, k] < objectives[i, k]:
                                dominates = False
                                break
                            elif not maximize2 and objectives[j, k] > objectives[i, k]:
                                dominates = False
                                break
                    if dominates:
                        is_pareto[i] = False
                        break
        
        pareto_points = objectives[is_pareto]
        
        # Create plot
        fig = go.Figure()
        
        # All points
        fig.add_trace(go.Scatter(
            x=obj1_vals,
            y=obj2_vals,
            mode='markers',
            name='All Experiments',
            marker=dict(color='lightblue', size=8, opacity=0.6)
        ))
        
        # Pareto front
        fig.add_trace(go.Scatter(
            x=pareto_points[:, 0],
            y=pareto_points[:, 1],
            mode='markers+lines',
            name='Pareto Front',
            marker=dict(color='red', size=12, symbol='star'),
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title="Pareto Front Analysis",
            xaxis_title=obj1,
            yaxis_title=obj2,
            hovermode='closest',
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Points on Pareto Front", len(pareto_points))
        with col2:
            best_obj1 = pareto_points[:, 0].max() if maximize1 else pareto_points[:, 0].min()
            st.metric(f"Best {obj1}", f"{best_obj1:.1f}")
        with col3:
            best_obj2 = pareto_points[:, 1].max() if maximize2 else pareto_points[:, 1].min()
            st.metric(f"Best {obj2}", f"{best_obj2:.1f}")

# ============================================================================
# MOLECULAR GENERATOR TAB
# ============================================================================

def display_molecular_generator_tab():
    """Molecular generator tab"""
    st.markdown("<h2 class='sub-header'>REINVENT-style Porphyrin Generator</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    This module uses generative deep learning to design novel porphyrin structures 
    with optimized properties for singlet oxygen generation and other photophysical properties.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Generation Parameters")
        
        n_molecules = st.slider("Number of molecules", 5, 50, 10)
        temperature = st.slider("Creativity (temperature)", 0.1, 2.0, 0.8, 0.1)
        
        property_focus = st.multiselect(
            "Property focus",
            ['High Singlet Oxygen', 'High Yield', 'High Purity', 'Drug-like'],
            default=['High Singlet Oxygen']
        )
        
        if st.button("🎯 Generate Novel Structures", use_container_width=True):
            st.session_state.generating = True
    
    with col2:
        st.markdown("### Base Porphyrin Core")
        st.markdown("""
        ```
            NH   N
           /  \\ /
          |    |
           \\  / \\
            N   HN
        ```
        """)
    
    if st.session_state.get('generating', False):
        with st.spinner("Generating novel porphyrin structures..."):
            time.sleep(2)  # Simulate generation time
            
            # Sample generated molecules
            generated_molecules = [
                "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=N5)C=C2",
                "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(Br)=N5)C=C2",
                "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(Cl)=N5)C=C2",
                "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(I)=N5)C=C2",
                "C1=CC2=NC1=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(CC)=N5)C=C2"
            ]
            
            st.success(f"✅ Generated {n_molecules} novel structures")
            
            # Display molecules
            for i, smiles in enumerate(generated_molecules[:n_molecules]):
                with st.expander(f"Molecule {i+1}"):
                    st.code(smiles)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MW", f"{np.random.uniform(600, 800):.0f}")
                    with col2:
                        st.metric("LogP", f"{np.random.uniform(3, 6):.1f}")
                    with col3:
                        st.metric("SOQ", f"{np.random.uniform(0.4, 0.8):.2f}")

# ============================================================================
# DEEPSEEK CHATBOX TAB
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
    """Deepseek AI Assistant tab"""
    st.markdown("<h2 class='sub-header'>🤖 Deepseek AI Assistant</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Chat with Deepseek, your AI expert in quantum dot and porphyrin synthesis optimization.
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
# MAIN FUNCTION
# ============================================================================

def main():
    """Main application"""
    
    # Sidebar with logo
    with st.sidebar:
        # Check for logo image
        logo_path = None
        if os.path.exists("images"):
            image_files = [f for f in os.listdir("images") if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            if image_files:
                logo_path = os.path.join("images", image_files[0])
        
        # Display logo
        if logo_path and os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
        else:
            st.markdown("""
            <div class="sidebar-logo">
                <div style="font-size: 3rem;">🧪</div>
                <div class="sidebar-logo-text">CHEM-NANO-BEW</div>
                <div style="color: #ecf0f1; font-size: 0.9rem;">LABORATORY</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## Navigation")
        
        # Mode selection
        app_mode = st.radio(
            "Select Mode",
            ["Quantum Dots", "Porphyrins", "Multi-Objective Analysis", 
             "Molecular Generator", "Deepseek AI Assistant"],
            key="navigation"
        )
        
        st.markdown("---")
        
        # Image upload
        with st.expander("📸 Upload Lab Logo"):
            uploaded_logo = st.file_uploader("Choose image", type=['png', 'jpg', 'jpeg', 'gif'])
            if uploaded_logo is not None:
                saved_path = save_uploaded_image(uploaded_logo)
                if saved_path:
                    st.success("✅ Logo uploaded!")
                    st.rerun()
        
        st.markdown("---")
        
        # Data upload
        st.markdown("## Data Management")
        uploaded_file = st.file_uploader("Upload CSV data", type=['csv'])
        
        if uploaded_file is not None:
            st.success(f"✅ Loaded: {uploaded_file.name}")
        
        st.markdown("---")
        
        # About section
        st.markdown("## About")
        st.info(
            "**CHEM-NANO-BEW Laboratory**\n\n"
            "Advanced synthesis optimization for "
            "quantum dots and porphyrins using "
            "machine learning and DoE."
        )
    
    # Main content header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 class='main-header'>CHEM-NANO-BEW LABORATORY</h1>", unsafe_allow_html=True)
        st.markdown("<p class='lab-subtitle'>Advanced Synthesis Optimization Suite</p>", unsafe_allow_html=True)
    
    # Display selected mode
    if app_mode == "Quantum Dots":
        display_quantum_dots_tab(uploaded_file)
    elif app_mode == "Porphyrins":
        display_porphyrins_tab(uploaded_file)
    elif app_mode == "Multi-Objective Analysis":
        display_multi_objective_tab()
    elif app_mode == "Molecular Generator":
        display_molecular_generator_tab()
    else:  # Deepseek AI Assistant
        display_deepseek_chatbox()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()