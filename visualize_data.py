# BUILD-IN LIBRARIES
from pathlib import Path 

# THIRD-PARTY LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

# USER-DEFINED LIBRARIES

# CONSTANT VARIABLES
PATH_DATA_RAW = 'data_folder/winequality-red-raw.csv'

PATH_ANALYSIS_OUTLIER = 'data_visualization/outlier.png'
PATH_ANALYSIS_FEATURES_VS_QUALITY = 'data_visualization/features_vs_quality.png'
PATH_ANALYSIS_QUALITY_COUNT = 'data_visualization/quality_count.png'
PATH_ANALYSIS_HEATMAP = 'data_visualization/heatmap.png'


def plotOutliers(df: pd.DataFrame) -> int:
    
    fig, axes = plt.subplots(4, 3, figsize=(12, 8))
    axes: np.ndarray[Axes]
    axes = axes.flatten()

    total_cell_detected = 0
    all_outlier_index = set()
    
    for col, ax in zip(df.columns, axes):
        ax: Axes = ax
        # calculate bounds
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        bound_lower = Q1 - 1.5 * IQR
        bound_upper = Q3 + 1.5 * IQR
        
        # find outliers
        outlier_index = df[(df[col] < bound_lower) | (df[col] > bound_upper)].index
        all_outlier_index.update(outlier_index)
        outlier_number = len(outlier_index)
        total_cell_detected += len(outlier_index)
        
        # plot
        ax.boxplot(df[col], vert=False)
        ax.grid(True)
        ax.set_title(f'{col}: {outlier_number}\n')
        
                
    # drop outlier
    # df.drop(index=all_outlier_index, inplace=True)
        
    fig.suptitle(f'\nOUTLIERS ACROSS FEATURES\n' +
                 f'Cells detected: {total_cell_detected}\n' +
                 f'Columns dropped: {len(all_outlier_index)}\n')
    plt.tight_layout()
    plt.savefig(generatePath(PATH_ANALYSIS_OUTLIER), dpi=600, bbox_inches='tight')
    print(f'OUTLIERS saved to: {generatePath(PATH_ANALYSIS_OUTLIER)}')
    plt.close()
    
    return len(all_outlier_index)
        

def plotFeaturesVsQuality(df: pd.DataFrame, file_name=PATH_ANALYSIS_FEATURES_VS_QUALITY) -> int:
    
    fig, axes = plt.subplots(3, 4, figsize=(10,8))
    axes: np.ndarray[Axes]
    axes = axes.flatten()
    
    qualities = df['quality'].unique()
    qualities = sorted(qualities)

    for col, ax in zip(df.columns, axes):
        ax: Axes
        mean_values = df.groupby("quality")[col].mean()
        
        # Plot bar chart
        c = plt.cm.coolwarm(np.linspace(0,1,len(mean_values)))
        ax.bar(mean_values.index, mean_values.values, color=c)
        ax.set_xticks(qualities)
        ax.set_title(col)
        
    plt.suptitle('\nFEATURES MEANS VALUE VS QUALITY\n' +
                 'x: Features | y: Qualities\n')
    plt.tight_layout()
    plt.savefig(generatePath(file_name), dpi=600, bbox_inches='tight')
    print(f'FEATURE VS QUALITY saved to: {generatePath(file_name)}')
    
    plt.close()

def plotQualityCount(df: pd.DataFrame, file_name=PATH_ANALYSIS_QUALITY_COUNT):
    plt.figure(figsize=(6, 6))
    
    qualities = df['quality'].unique()
    qualities = sorted(qualities)
    
    count = []
    for q in qualities:
        matching_row = df[df['quality'] == q]
        c_matching_row = len(matching_row)
        count.append(c_matching_row)
        
    plt.bar(qualities, count)
    plt.title('Count of Wine Quality Level')
    plt.xlabel('Quality')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(generatePath(PATH_ANALYSIS_QUALITY_COUNT), dpi=600, bbox_inches='tight')
    print(f'QUALITY COUNT saved to: {generatePath(file_name)}')
    plt.close()
    
def plotCorrelationMatrix(df: pd.DataFrame, file_name=PATH_ANALYSIS_HEATMAP):
    
    plt.figure(figsize=(12, 12))
    correlation = df.corr()
    
    im = plt.imshow(correlation, cmap='Purples', interpolation='nearest')
    plt.colorbar(im)
    
    plt.xticks(range(len(correlation.columns)), df.columns, rotation=45)
    plt.yticks(range(len(correlation.columns)), df.columns)
    
    # fill heatmap
    for pos_y in range(len(correlation.columns)):
        for pos_x in range(len(correlation.columns)):
            plt.text(pos_x, pos_y,
                     f'{correlation.iloc[pos_y, pos_x]:.3f}',
                     ha='center', va='center')
    
    plt.title('\nCORRELATION HEATMAP')
 
    plt.tight_layout()
    plt.savefig(generatePath(file_name), dpi=600, bbox_inches='tight')
    plt.close()
    print(f'CORRELATION HEATMAP saved to: {generatePath(file_name)}')
            
     

def generatePath(file_name: str) -> str:
    # get current path and concatnate with parameter
    base = Path(__file__).resolve().parent
    path = base / file_name
    return path
    

if __name__ == '__main__':
    df = pd.read_csv(generatePath(PATH_DATA_RAW))
    plotQualityCount(df)    
    plotFeaturesVsQuality(df)
    plotCorrelationMatrix(df)
    plotOutliers(df)

    
    