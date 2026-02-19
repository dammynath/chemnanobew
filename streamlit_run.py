# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 08:55:36 2026

@author: NATHANAEL
"""

import subprocess

file = "app.py"
#file = "app_plots.py"
#file = "app_profiler.py"
#file = "app_profiler_menus.py"
#file = "Streamlit_day3.py"
#file = "app.py"


subprocess.Popen(
    ["streamlit", "run", file], shell=True
)

