""" Main file to run seasonal and monthly forecast"""
# Current scripts directory
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor,LinearRegression
from class_regression_v2 import RegressionModel



directory: str = "/Users/fco/CAPTA/Pronostico_estacional_Python"
os.chdir(directory)


def run_model(month_initialisation, cod_cuenca):
        """
        @Parameters
        X : predictors. e.g. pp_mm cumsum, tem_C avg, soil_moisture_mm cumsum, snow_swe avg, etc
        y : cumulative volume in a period. e.g. vol_mm sep-mar, jan-mar, etc
        q : average monthly streamflows in a period (same as volume). e.g q_mm sep, q_mm oct... q_mm mar
        """
        ## Step 1: import and split data (predictors, volume)
        model = RegressionModel(month_initialisation = month_initialisation,
                                window_strategy = 'dynamic',
                                catchment_code= cod_cuenca,
                                predictor_list=["pr_sum_-1months","tem_mean_3months"],
                                wy_holdout=2016)
        
        ## Step 2: compute determinist and ensemble forecast (regression model)
        model.set_model_parameters(model_scaler = StandardScaler(),
                                   model_regressor= LinearRegression())
        
        model.forecast_vol_ensemble(n_ens_members=1000)
        ## Step 3: plot ensemble forecast
        #model.subplot_boxplot_chronological_median_order(export=False, subplot=False)
        ## Step 4: compute monthly streamflows based on closest predictors from previous years
        #model.forecast_q_ensemble(n_neighbors=10,
        #                          weights='distance',
        #                          p=2)
        
        ## Step 5: plot ensemble flow forecast
        #model.plot_knn_flow(export=False)
        
        #model.save_modelinfo()
        
        return model

#if __name__ == '__main__':


model_info=run_model("oct",'5410002')
y_train = model_info.y_train
y_cv = model_info.y_cv
y_fore = model_info.y_fore
rmse_cv = model_info.rmse_cv
self = model_info

    
        