# BUILD-IN LIBRARIES
from pathlib import Path 

# THIRD-PARTY LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

# USER-DEFINED LIBRARIES
from preprocessing import (
    Path,
    PATH_DATA_PREPROCESSED,
    generatePath
)


# CONSTANT VARIABLES
PATH_ANALYSIS_FEATURE_VS_TARGER = 'analysis/features_vs_quality.csv'


# PLOTTING
def plotCountQuality(df: pd.DataFrame, columns, ):
    pass
    
def dropIrrelevantFeatures(df: pd.DataFrame, columns: list[str], file_name=PATH_ANALYSIS_FEATURE_VS_TARGER) -> int:
    
    fig, axes = plt.subplots(3, 4, figsize=(10,8))
    axes: np.ndarray[Axes]
    axes = axes.flatten()
    
    for col, ax in zip(columns, axes):
        ax: Axes
        ax.scatter(df[col], df['quality'], s=5)
        ax.set_title(f'{col}')
        
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    df = pd.read_csv(generatePath(PATH_DATA_PREPROCESSED))
    

    
    