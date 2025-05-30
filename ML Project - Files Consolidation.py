#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 13:56:29 2025

@author: mariamhassan
"""
from github import Github
import pandas as pd

# Initialize GitHub connection
g = Github()
repo = g.get_repo("Parth-0804/ML_Presentation_Workspace")
contents = repo.get_contents("Europark_dataset", ref="mariam")

# Initialize empty DataFrames
fjord_rafting_df = pd.DataFrame()
eurosat_df = pd.DataFrame()
eurmir_df = pd.DataFrame()

# Loop through all files in the folder
for file in contents:
    if file.name.endswith(".xlsx"):
        file_name = file.name
        url = file.download_url

        try:
            # Read Excel file
            df = pd.read_excel(url)

            # Extract attraction name from the filename
            attraction_name = file_name.replace(" - Queue Times.xlsx", "")
            df["attraction_name"] = attraction_name

            # Assign to appropriate DataFrame
            if "Fjord Rafting" in file_name:
                fjord_rafting_df = pd.concat([fjord_rafting_df, df], ignore_index=True)
            elif "Eurosat - CanCan Coaster" in file_name:
                eurosat_df = pd.concat([eurosat_df, df], ignore_index=True)
            elif "Euro-Mir" in file_name or "Euromir" in file_name:
                eurmir_df = pd.concat([eurmir_df, df], ignore_index=True)

        except Exception as e:
            print(f"Error reading {file.name}: {e}")

# Preview the results
print("🎢 Fjord Rafting:")
print(fjord_rafting_df.head())

print("\n🎢 Eurosat - CanCan Coaster:")
print(eurosat_df.head())

print("\n🎢 Euro-Mir:")
print(eurmir_df.head())
