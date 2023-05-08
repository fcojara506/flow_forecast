from default_input_parameters import attributes_catchments
from main_regression_model_class import run_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from default_input_parameters import months_initialisation, cod_cuencas

def main():
    # loop through model parameters
    model_info = dict()
   
    i = 0
    for month_initialisation in months_initialisation:
        for cod_cuenca in cod_cuencas:
            model_info[i] = run_model(month_initialisation,cod_cuenca).info
            i += 1
    return model_info

def reorganise_data(model_info):
    data = dict()
    for _,v_out in model_info.items():
        for k , v in v_out.items():
            if k in data:
                data[k].append(v)
            else:
                data[k] = [v]
    return data

## model data from iterations
model_info = reorganise_data(main())
df_model_info = pd.DataFrame(model_info).drop(['scores_vol'],axis=1)

scores_vol = pd.concat(model_info['scores_vol'],ignore_index=True).rename_axis('id')
info_vol = pd.concat([df_model_info,scores_vol],axis=1)

## catchment data
attributes_catchments_=attributes_catchments[["cod_cuenca","gauge_lat","gauge_lon","mean_elev"]]

## merge data
info_vol_catchments = pd.merge(info_vol,attributes_catchments_,right_on='cod_cuenca',left_on='catchment_code')


fig, axs = plt.subplots(1, figsize=(7.5, 4),dpi=600)
g = sns.lineplot(
    x='month_initialisation',
    y='crpss_climatology',
    hue = 'mean_elev',
    data = info_vol_catchments,
    palette= "viridis",
    #marker = 's'
    )
axs.set_ylim(top=1)
plt.axhline(y=0,color='black')

plt.title("Forecast CRPS Skill" )
plt.xlabel("Month of initialisation of the forecast")
plt.ylabel("CRPSS (reference: climatology)")
plt.legend(frameon=False,title="Median elevation (masl)",loc="upper left")

plt.savefig("data_output/pronostico_volumen/Figures/metrics/CRPSS_skill.png", bbox_inches='tight')

fig, axs = plt.subplots(1, figsize=(4, 7.5),dpi=600)
g = sns.scatterplot(
    y='gauge_lat',
    x='crpss_climatology',
    hue = 'month_initialisation',
    data = info_vol_catchments,
    palette= "viridis",
    #marker = 's'
    )

plt.xlabel("CRPSS (reference: climatology)")
plt.ylabel("Catchment latitude (degrees)")
plt.title("CRPS Skill (reference: climatology)")
#plt.legend(frameon=False,title="Median elevation\n (masl)",loc="best")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1),
          ncol=4, fancybox=True, shadow=True)

plt.savefig("data_output/pronostico_volumen/Figures/metrics/CRPSS_gaugelat.png", bbox_inches='tight')

