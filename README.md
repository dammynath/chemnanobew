# chemnanobew
CHEMNANOBEW AUTOMATION PAGE

# Automated DoE Synthesis Optimizer
### CIS-Te/ZnS Quantum Dots & Porphyrins for Phototherapy

This Streamlit application provides an **automated Design of Experiments (DoE) + machine learning workflow** for optimizing:

• CIS-Te/ZnS quantum dots (NIR photoluminescence)  
• Porphyrins (photothermal + photodynamic therapy performance)

The goal is to **replace manual trial-and-error synthesis** with data-driven predictions of optimal reaction conditions.

---

## Why this app?

Phototherapy materials require:

### Quantum dots
- Emission ≥ 800 nm (NIR window)
- High photoluminescence intensity
- Stable, non-aggregated cores/shells

### Porphyrins
- Absorption ≥ 800 nm
- Photothermal conversion efficiency (PCE) > 60%
- High singlet oxygen (¹O₂) generation

Instead of testing dozens of syntheses manually, this app:

1. Generates a DoE matrix
2. Trains a regression model
3. Predicts optimal conditions
4. Suggests next experiments automatically

---

# App Structure

The app has **two tabs**:

## Tab 1 – CIS-Te/ZnS Quantum Dots

### Inputs
- pH
- Tellurium amount (g)
- Zinc acetate amount (g)
- Shell growth
