# Sources: https://www.kaggle.com/code/ibrahimqasimi/wine-chemistry-to-quality-ml-guide/notebook 
# BUILD-IN LIBRARIES

# THIRD-PARTY LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split

# USER-DEFINED LIBRARIES
from visualize_data import generatePath, PATH_DATA_RAW

# CONSTANT VARIABLES
PATH_DATA_PREPROCESSED = 'data_folder/winequality-red-preprocessed.csv'

def preprocessData():
    
    df_preprocessed = pd.read_csv(generatePath(PATH_DATA_RAW))
    
    # Acidity balance features
    df_preprocessed['total_acidity'] = df_preprocessed['fixed acidity'] + df_preprocessed['volatile acidity'] + df_preprocessed['citric acid']
    df_preprocessed['acidity_ratio'] = df_preprocessed['fixed acidity'] / (df_preprocessed['volatile acidity'] + 0.01)
    
    # # Sulfur dioxide features
    df_preprocessed['free_sulfur_ratio'] = df_preprocessed['free sulfur dioxide'] / (df_preprocessed['total sulfur dioxide'] + 0.01)
    df_preprocessed['sulfur_dioxide_diff'] = df_preprocessed['total sulfur dioxide'] - df_preprocessed['free sulfur dioxide']
    
    # Sugar–alcohol interaction
    df_preprocessed['sugar_alcohol_ratio'] = df_preprocessed['residual sugar'] / (df_preprocessed['alcohol'] + 0.01)
    df_preprocessed['sweetness_index'] = df_preprocessed['residual sugar'] * (1 - df_preprocessed['alcohol'] / 100)
    
    # Mineral content indicators
    df_preprocessed['mineral_content'] = df_preprocessed['chlorides'] + df_preprocessed['sulphates']
    
    # pH–acidity relationship
    df_preprocessed['ph_acidity_interaction'] = df_preprocessed['pH'] * df_preprocessed['total_acidity']
    
    # Wine body indicator (density–alcohol relationship)
    df_preprocessed['wine_body'] = df_preprocessed['density'] * df_preprocessed['alcohol']
    
    # Round data
    df_preprocessed = df_preprocessed.round(5)
    
    # Target Conversion (3 classes only)
    def categorize_quality(q: pd.Series):
        if q <= 4:
            return 0  # Poor
        elif q <= 6:
            return 1  # Average
        else:
            return 2  # Excellent

    df_preprocessed['quality_class'] = df_preprocessed['quality'].apply(categorize_quality)
    df_preprocessed.drop(columns='quality', inplace=True)
    df_preprocessed.to_csv(generatePath(PATH_DATA_PREPROCESSED), index=False)
    
    print(f'Done preproccessing raw data. Saved to {generatePath(PATH_DATA_PREPROCESSED)}')
    

if __name__ == '__main__':
    preprocessData()