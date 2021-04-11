import numpy as np
import pandas as pd
from RegscorePy import aic, bic
from ctgan import CTGANSynthesizer

data = pd.read_csv("data/Data-Flood.csv")

continuous_columns = list(data.columns)  #selects column names only

ctgan = CTGANSynthesizer()
ctgan.fit(data, continuous_columns, epochs=10) # default is 300 epochs

# Create synthetic data for x number of rows
samples = ctgan.sample(207)

# Save synthetic database to csv
samples.to_csv('output/syntheticdata.csv',index=False)

X = np.genfromtxt('data/Data-Flood.csv', delimiter = ',')
new_X = np.delete(X, 0)

Y = np.genfromtxt('output/syntheticdata.csv', delimiter = ',')
new_Y = np.delete(Y, 0)

# Akaike's Information Criteria
print("AIC:", aic.aic(new_X, new_Y, 4))

# Bayesian Information Criteria
print("BIC: ", bic.bic(new_X, new_Y, 4))