# BUILD-IN LIBRARIES
import os 

# THIRD-PARTY LIBRARIES
import pandas as pd
import joblib as jl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler

# USER-DEFINED LIBRARIES
from visualize_data import generatePath
from preprocess_data import PATH_DATA_PREPROCESSED

# CONSTANTS VARIABLES
RANDOM_STATE = 66
PATH_MODEL = 'models/'

def train_models(X_train, X_train_scaled, y_train):
    
    models = {
        'random_forests': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=200, 
            min_samples_leaf=4,
            min_samples_split=5),
        'gradient_boost': GradientBoostingClassifier(
            random_state=RANDOM_STATE),
        'logistic_regression': LogisticRegression(
            random_state=RANDOM_STATE),
        'knn': KNeighborsClassifier(n_neighbors=10)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        if name in ['logistic_regression', 'knn']:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        
        trained_models[name] = model
        save_model(model, generatePath(PATH_MODEL + name + '.joblib'))
        
    return trained_models
        

# e. SAVE MODEL TO FILE
def save_model(model, path):
    jl.dump(model, path)
    print(f'Model saved to {path}')
    
# f. LOAD MODEL FROM FILE
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'No model found at {path}')
    model = jl.load(path)
    print(f'Model loaded from {path}')
    return model

def main():
    df = pd.read_csv(generatePath(PATH_DATA_PREPROCESSED))
    X = df.drop(columns='quality_class')
    y = df['quality_class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = train_models(X_train, X_train_scaled, y_train)
    for name, model in models.items():
        if name in ['logistic_regression', 'knn']:
            score = model.score(X_test_scaled, y_test)
        else:
            score = model.score(X_test, y_test)
        print(f'Score {name}: {score:.4f}')



if __name__ == '__main__':
    main()