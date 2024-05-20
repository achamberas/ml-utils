'''
Configuration file containing different settings for transformation functions, models and grid search.
'''

# define transformation functions
TRANSFORM_FUNCTIONS = {
    "ln": {
        "direct": "np.log2",
        "inverse": "np.exp2"
    },
    "square": {
        "direct": "np.square",
        "inverse": "np.sqrt"
    },
    "sqrt": {
        "direct": "np.sqrt",
        "inverse": "np.square"
    }
}

# define default random forrest regressor params
DEFAULT_RandomForestRegressor_HP = {
    "bootstrap": False,
    "max_depth":20,
    "max_features": "sqrt",
    "min_samples_leaf" :2,
    "min_samples_split": 10,
    "n_estimators": 80   
}

# define default grid search params
DEFAULT_GRID_SEARCH = {
    "param_distributions": {
        "n_estimators": [10,20,30,40,50,60,70,80,90,100],       # num of trees in forest
        "max_features": ['sqrt', 'log2', None],                 # num features @ split
        "max_depth": [10,20,30,40,50,60,70,80,90,100, None],    # Max num of levels in tree
        "min_samples_split": [2, 5, 10],                        # Min num samples to split node
        "min_samples_leaf": [1, 2, 4],                         # Min num samples required @ leaf node
        "bootstrap": [True, False]                              # sample selection method
    },
    "n_iter": 100,
    "cv": 5,
    "verbose": 0,
    "random_state": 42,
    "n_jobs": -1
}

# define overall default params
CONFIG = {
    "experiment_name": "Test",
    "experiment_desc": "Test",
    "training_data_simid": ['20240119-214210-EQHI1', '5678'],
    "climate_zones": ["5A", "5B", "5C", "6A", "6B"],
    "target": "total_eui_elec_gj_m2",
    "transformation": "ln",
    "drop_cols": ["ESA", "AR", "HD", "ahu_burner_efficiency", "supply_air_temp_heating", "temp_setpoint_heating_occupied", "temp_setpoint_heating_setback", "total_eui_ng_gj_m2", "total_eui_elec_gj_m2", "total_eui_elec_gj_m2_ln"],
    "minority_pct": 0.5,
    "random_seed": 100,
    "model": {
        "algorithm": "RandomForestRegressor",
    },
    "error_analysis_simid": ['4321']
}