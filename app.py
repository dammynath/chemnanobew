# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 08:48:40 2026

@author: NATHANAEL
"""
"""
Synthesis_app/
"""

import streamlit as st

#st.set_page_config(layout="wide")   # MUST be first Streamlit command
#st.set_page_config(page_title="Synthesis App", layout="wide")
st.title("CHEMNANOBEW App")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

#st.set_page_config(layout="wide")

############################################
# Utility functions
############################################

def generate_doe(ranges, n_runs=30):
    data = {}
    for k, (low, high) in ranges.items():
        data[k] = np.random.uniform(low, high, n_runs)
    return pd.DataFrame(data)


def train_model(df, target_cols):
    X = df.drop(columns=target_cols)
    models = {}
    for t in target_cols:
        y = df[t]
        model = RandomForestRegressor(n_estimators=200)
        model.fit(X, y)
        models[t] = model
    return models


def predict_optimum(models, ranges, objective_fn, samples=3000):
    grid = generate_doe(ranges, samples)
    scores = []

    for i in range(len(grid)):
        row = grid.iloc[i:i+1]
        preds = {k: models[k].predict(row)[0] for k in models}
        score = objective_fn(preds)
        scores.append(score)

    best = grid.iloc[np.argmax(scores)]
    return best


############################################
# APP HEADER
############################################

st.title("Automated DoE Synthesis Optimizer")
tabs = st.tabs(["CIS-Te/ZnS Quantum Dots", "Porphyrins"])


##################################################################
# TAB 1 – QUANTUM DOT WORKFLOW
##################################################################

with tabs[0]:

    st.header("CIS-Te/ZnS Quantum Dot DoE Optimization")

    st.write("### Define synthesis ranges")

    pH = st.slider("pH", 2.5, 6.0, (3.0, 5.0))
    Te = st.slider("Tellurium (g)", 0.0012, 0.0022, (0.0016, 0.0020))
    Zn = st.slider("Zinc acetate (g)", 0.020, 0.040, (0.025, 0.035))
    shell = st.slider("Shell time (min)", 10, 60, (15, 45))
    ligand = st.selectbox("Ligand", ["TGA", "MPA"])

    runs = st.slider("Number of DOE experiments", 10, 80, 30)

    if st.button("Generate DOE Plan"):

        ranges = {
            "pH": pH,
            "Te": Te,
            "Zn": Zn,
            "shell_time": shell
        }

        df = generate_doe(ranges, runs)

        # simulate expected responses using known experimental trends
        df["wavelength"] = (
            720
            + 250*df["Te"]
            - 6*(df["pH"]-4)**2
            + 0.4*df["shell_time"]
        )

        df["intensity"] = (
            15000
            + 4000*(df["pH"]-4)
            - 200*(df["shell_time"]-25)**2/50
            + np.random.normal(0, 1000, runs)
        )

        st.dataframe(df)

        st.download_button(
            "Download DOE CSV",
            df.to_csv(index=False),
            "QD_DOE_plan.csv"
        )

        st.session_state.qd_df = df

    ####################################
    # MODEL + OPTIMIZER
    ####################################

    if "qd_df" in st.session_state:

        st.subheader("Train model + predict optimum")

        df = st.session_state.qd_df

        models = train_model(df, ["wavelength", "intensity"])

        def objective(preds):
            # maximize both wavelength and intensity
            return preds["wavelength"] + preds["intensity"]/1000

        ranges = {
            "pH": pH,
            "Te": Te,
            "Zn": Zn,
            "shell_time": shell
        }

        best = predict_optimum(models, ranges, objective)

        st.success("Predicted optimum conditions")

        st.write(best)

        fig = px.scatter(
            df,
            x="wavelength",
            y="intensity",
            title="QD DOE Response Space"
        )
        st.plotly_chart(fig)



##################################################################
# TAB 2 – PORPHYRIN WORKFLOW
##################################################################

with tabs[1]:

    st.header("Porphyrin Phototherapy DoE Optimization")

    st.write("### Define synthesis variables")

    metal = st.selectbox("Metal center", ["None", "Zn", "Cu", "Pd"])

    temp = st.slider("Reaction temp (°C)", 50, 180, (90, 150))
    time = st.slider("Reaction time (h)", 1, 24, (6, 18))
    substituent = st.slider("Electron withdrawing index", 0.0, 1.0, (0.2, 0.8))

    runs = st.slider("DOE runs", 10, 80, 30, key="porph_runs")

    if st.button("Generate Porphyrin DOE"):

        ranges = {
            "temp": temp,
            "time": time,
            "substituent": substituent
        }

        df = generate_doe(ranges, runs)

        # empirical phototherapy relations
        df["absorption"] = 650 + 250*df["substituent"] + 0.2*df["temp"]
        df["PCE"] = 40 + 30*df["substituent"]
        df["singletO2"] = 0.4 + 0.6*df["substituent"]

        st.dataframe(df)

        st.session_state.porph_df = df

    if "porph_df" in st.session_state:

        df = st.session_state.porph_df

        models = train_model(df, ["absorption", "PCE", "singletO2"])

        def objective(preds):
            return preds["absorption"] + preds["PCE"] + 100*preds["singletO2"]

        ranges = {
            "temp": temp,
            "time": time,
            "substituent": substituent
        }

        best = predict_optimum(models, ranges, objective)

        st.success("Predicted optimal porphyrin synthesis")

        st.write(best)

        fig = px.scatter(
            df,
            x="absorption",
            y="PCE",
            title="Porphyrin DOE"
        )
        st.plotly_chart(fig)



##########################################################
# FOOTER
##########################################################

st.info("Upload real experimental results to replace simulated responses for true optimization.")
