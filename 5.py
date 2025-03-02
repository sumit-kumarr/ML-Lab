import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load the dataset
heartDisease = pd.read_csv('heart.csv')  # Ensure the file path is correct
heartDisease = heartDisease.replace('?', np.nan)

# Check column names
print('Dataset Columns:', heartDisease.columns)

# Print dataset details
print('Sample instances from the dataset:')
print(heartDisease.head())
print('\nAttributes and datatypes:')
print(heartDisease.dtypes)

# Define the Bayesian Model (Update 'cp' if needed)
model = BayesianModel([
    ('age', 'HeartDisease'),
    ('sex', 'HeartDisease'),
    ('exang', 'HeartDisease'),
    ('cp', 'HeartDisease'),  # Ensure this matches the dataset column name
    ('HeartDisease', 'restecg'),
    ('HeartDisease', 'chol')
])

# Learn CPD using Maximum Likelihood Estimators
print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Perform inference
print('\nInferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)

# Query 1: Probability of HeartDisease given evidence= restecg
print('\n1. Probability of HeartDisease given evidence= restecg')
q1 = HeartDisease_infer.query(variables=['HeartDisease'], evidence={'restecg': 1})
print(q1)

# Query 2: Probability of HeartDisease given evidence= cp
print('\n2. Probability of HeartDisease given evidence= cp')
q2 = HeartDisease_infer.query(variables=['HeartDisease'], evidence={'cp': 2})
print(q2) 
