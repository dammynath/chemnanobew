# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 08:55:36 2026

@author: NATHANAEL
"""

import subprocess

#file = deep.py
file = "chemnanobew_app.py"
#file = "new_app.py"

subprocess.Popen(
    ["streamlit", "run", file], shell=True
)



