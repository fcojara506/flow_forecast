""" Main file to run seasonal and monthly forecast"""
# Current scripts directory
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import LeaveOneOut
from class_regression import RegressionModel


directory: str = "/Users/fco/CAPTA/Pronostico_estacional_Python"
os.chdir(directory)

# loop through model parameters
months_initialisation = ['may','jun','jul','ago','sep','oct','nov','dic','ene','feb','mar']

for month_initialisation in months_initialisation:
    """
    @Parameters
    X : predictors. e.g. pp_mm cumsum, tem_C avg, soil_moisture_mm cumsum, snow_swe avg, etc
    y : cumulative volume in a period. e.g. vol_mm sep-mar, jan-mar, etc
    q : average monthly streamflows in a period (same as volume). e.g q_mm sep, q_mm oct... q_mm mar
    """
    ## Step 1: import and split data (predictors, volume)
    model = RegressionModel(month_initialisation = month_initialisation,
                             catchment_code= "5410002",
                             predictor_list=["pr_sum_-1months"],
                             wy_holdout=1985)
    ## Step 2: compute determinist and ensemble forecast (regression model)
    model.set_model_parameters(model_scaler = StandardScaler(),
                               model_regressor= HuberRegressor(),
                               cv = LeaveOneOut())
    model.forecast_vol_ensemble(n_ens_members=1000)
    ## Step 3: plot ensemble forecast
    model.subplot_boxplot_chronological_median_order(export=False, subplot=False)
    ## Step 4: compute monthly streamflows based on closest predictors from previous years
    model.forecast_q_ensemble(num_neighbors=10,
                              type_weights='distance',
                              metric="wminkowski",
                              p=2)
    ## Step 5: plot ensemble flow forecast
    model.plot_knn_flow(export=False)
    break
    