"""Importing the libraries that are used in the script."""
import pyarrow.feather as feather
import pandas as pd



def month_to_wym(month):
    wym = int(month)
    wym = wym-3 if wym>3 else wym+12-3
    #wym = "{0:02d}".format(wym)
    return wym

def year_to_wy(year,month):
    year = int(year)
    month = int(month)
    wy = year if month>3 else year-1
    return wy

pr_filename = "data_input/era5_pr_daily.feather"
tem_filename = "data_input/era5_tem_daily.feather"

pr_daily = feather.read_feather(pr_filename)
tem_daily = feather.read_feather(tem_filename)

pr_monthly = pr_daily.groupby(["month","year"]).sum().reset_index().melt(id_vars=["month","year"],
                                                                         var_name="cod_cuenca",
                                                                         value_name="pr")

tem_monthly = tem_daily.groupby(["month","year"]).mean().reset_index().melt(id_vars=["month","year"],
                                                                         var_name="cod_cuenca",
                                                                         value_name="tem")
meteo_monthly = pd.merge(pr_monthly,tem_monthly)
meteo_monthly["wym"] = meteo_monthly.month.apply(lambda x: month_to_wym(x))
meteo_monthly["wy_simple"] = meteo_monthly.apply(lambda x: year_to_wy(x.year, x.month), axis=1)

meteo_filename = "data_input/predictors_monthly_catchments_ChileCentral_era5.feather"
feather.write_feather(meteo_monthly, meteo_filename)


###

flow_data = "data_input/flows_mm_monthly_catchments_ChileCentral.feather"
a = feather.read_feather(flow_data)