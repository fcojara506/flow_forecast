## Long term Forecast of flow and its volume

# flow forecast
the streamflow is derived from the volume using kNN and the same predictors used for the regressions.
In this the method of regression doesn't affect significally the resultant error (see some metrics below)


The forecast ensembles spread change along the year for different initialisation times.

1st MAY
![snow](https://github.com/fcojara506/flow_forecast/blob/main/data_output/pronostico_caudal/Figures/ensemble_forecast/5410002/flow_ensemble_forecast_5410002_1stmay_pr_sum_-1months_AND_tem_mean_3months_%5Boct%2Cmar%5D_2013.png)

1st JULY
![snow](https://github.com/fcojara506/flow_forecast/blob/main/data_output/pronostico_caudal/Figures/ensemble_forecast/5410002/flow_ensemble_forecast_5410002_1stjul_pr_sum_-1months_AND_tem_mean_3months_%5Boct%2Cmar%5D_2013.png)

1st SEPTEMBER
![snow](https://github.com/fcojara506/flow_forecast/blob/main/data_output/pronostico_caudal/Figures/ensemble_forecast/5410002/flow_ensemble_forecast_5410002_1stsep_pr_sum_-1months_AND_tem_mean_3months_%5Boct%2Cmar%5D_2013.png)

1st NOVEMBER
![snow](https://github.com/fcojara506/flow_forecast/blob/main/data_output/pronostico_caudal/Figures/ensemble_forecast/5410002/flow_ensemble_forecast_5410002_1stnov_pr_sum_-1months_AND_tem_mean_3months_%5Bnov%2Cmar%5D_2013.png)




# Volume

1st MAY
![snow](https://github.com/fcojara506/flow_forecast/blob/main/data_output/pronostico_volumen/Figures/ensemble_forecast/5410002/EnsembleVolumeHindcast_5410002_1stmay_pr_sum_-1months_AND_tem_mean_3months_%5Boct%2Cmar%5D2013.png)

1st JULY
![snow](https://github.com/fcojara506/flow_forecast/blob/main/data_output/pronostico_volumen/Figures/ensemble_forecast/5410002/EnsembleVolumeHindcast_5410002_1stjul_pr_sum_-1months_AND_tem_mean_3months_%5Boct%2Cmar%5D2013.png)

1st SEPTEMBER
![snow](https://github.com/fcojara506/flow_forecast/blob/main/data_output/pronostico_volumen/Figures/ensemble_forecast/5410002/EnsembleVolumeHindcast_5410002_1stsep_pr_sum_-1months_AND_tem_mean_3months_%5Boct%2Cmar%5D2013.png)

1st NOVEMBER
![snow](https://github.com/fcojara506/flow_forecast/blob/main/data_output/pronostico_volumen/Figures/ensemble_forecast/5410002/EnsembleVolumeHindcast_5410002_1stnov_pr_sum_-1months_AND_tem_mean_3months_%5Bnov%2Cmar%5D2013.png)


# metrics

Continous Ranked Probability Skill Score (skill for many predictions)
![snow](https://github.com/fcojara506/flow_forecast/blob/main/data_output/pronostico_volumen/Figures/metrics/CRPSS_skill.png)

CRPSS with latitude and month of initialisation
![snow](https://github.com/fcojara506/flow_forecast/blob/main/data_output/pronostico_volumen/Figures/metrics/CRPSS_gaugelat.png)


percentage bias 
![snow](https://github.com/fcojara506/flow_forecast/blob/main/data_output/pronostico_volumen/Figures/metrics/5410002/Pbias_predictors-models_test_jun_5410002.png)

root mean square error
![snow](https://github.com/fcojara506/flow_forecast/blob/main/data_output/pronostico_volumen/Figures/metrics/5410002/RMSE_predictors-models_test_jun_5410002.png)


scatter plot of observed vs predicted
![snow](https://github.com/fcojara506/flow_forecast/blob/main/data_output/pronostico_volumen/Figures/metrics/5410002/Scatter_Xtrain_ypred_predictors-models_test_sep_prAcum_5410002.png)



