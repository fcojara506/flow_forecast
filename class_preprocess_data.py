"""Importing the libraries that are used in the script."""
import pyarrow.feather as feather
from dataclasses import dataclass
import pandas as pd


@dataclass
class CatchmentData:
    """Extract meteo and hydrological data for one specific river catchment"""
    catchment_code: str
    catchments_attributes_filename: str = "data_input/attributes_catchments_ChileCentral.feather"
    flows_filename: str = "data_input/flows_mm_monthly_catchments_ChileCentral.feather"
    meteo_filename: str = "data_input/predictors_monthly_catchments_ChileCentral.feather"
    hydro_filename: str = "data_input/hydro_variables_monthly_catchments_ChileCentral.feather"
    
    def __post_init__(self):
        self._read_catchment_data()
        self._define_input_data_predictors()
    
    def _set_water_year_months(self):
        self.months_wy = ['abr','may','jun','jul','ago','sep','oct','nov','dic','ene','feb','mar']
        
    def _read_filter_feather(self,filename):
        "Read feather data and filter by catchment"
        df = feather.read_feather(filename)
        df = df[df.cod_cuenca == self.catchment_code]
        df = df.drop(['cod_cuenca'],axis=1)
        return df
     
    @staticmethod
    def _columns_to_int(df):
        """Format wy-months and wy columns to integers"""
        df.wym = df.wym.astype(int)
        df.wy_simple = df.wy_simple.astype(int)
        return df
        
    def _read_catchment_data(self) -> None:
        """Filter the data only to the selected catchment."""
        self.gauge_name = self._read_filter_feather(self.catchments_attributes_filename)["gauge_name"].to_string(index=False)
        self.monthly_meteo = self._columns_to_int(self._read_filter_feather(self.meteo_filename))
        self.monthly_flows = self._columns_to_int(self._read_filter_feather(self.flows_filename))
    
    def _define_input_data_predictors(self):
        """ Choose which dataframes to input in predictors"""
        self.raw_data_df = self.monthly_meteo
     
          
@dataclass   
class ForecastWindow:
    """ Define a dynamic window between two target month of forecast"""
    month_initialisation : str
    month_init_fixed: str = "oct"
    month_end_fixed: str = "mar"
    
    
    def __post_init__(self):
        
        self._set_water_year_months()
        self._set_month_initialisation_index()
        self._set_window_indexes()
        self._set_forecast_period()
    
    def _set_month_initialisation_index(self) -> None:
        """Set the index of month_initialisation in the water year"""
        self.month_initialisation_index = self.months_wy.index(self.month_initialisation)
        
    def _set_window_indexes(self) -> None:
        self.init_target_index = self.months_wy.index(self.month_init_fixed)
        self.end_target_index = self.months_wy.index(self.month_end_fixed)

    def _set_forecast_start_month(self) -> None:
        """Start month of target forecast period (init_forecast_index)"""
        if self.window_strategy == 'dynamic':
            "before init_target_index fixed method then dynamic"
            if self.month_initialisation_index <= self.init_target_index:
                self.init_forecast_index = self.init_target_index 
            else:
                self.init_forecast_index = self.month_initialisation_index
        elif self.window_strategy == 'fixed':
            self.init_forecast_index = self.init_target_index
    
    def _set_forecast_end_month(self) -> None:
        """End month of target forecast period (end_forecast_index)"""
        if self.window_strategy == 'dynamic':
            self.end_forecast_index = self.end_target_index
        elif self.window_strategy == 'fixed':
            self.end_forecast_index = self.end_target_index
    
    def _set_forecast_period(self) -> None:
        """List of months in the forecast period"""
        self._set_forecast_start_month()
        self._set_forecast_end_month()
        self.months_forecast_period = self.months_wy[self.init_forecast_index:self.end_forecast_index+1]
        self.months_before_initialisation = self.months_wy[0:self.month_initialisation_index]



@dataclass
class Predictors:
    """Generate predictors from meteo and hydro data"""
    predictor_list : list
    
    def __post_init__(self):
        self._get_predictors()
    #    self.raw_data_df = self.raw_data.raw_data_df 
    #    self.month_initialisation_index = self.dates_data.month_initialisation_index
        
    def _separate_variables(self,var_name):
        "Separate var_name into variable, operator and period"
        self.variable,self.operator,period_before = var_name.split('_')
        self.period_before = int(period_before.removesuffix('months'))
        
    def _check_period(self):
        "Check if var_name contains negative number of months"
        if self.period_before < 0:
            self.period_before = self.month_initialisation_index        
            
    def _rename_varname(self):
        "Rename var_name in case some variables changed"
        return f"{self.variable}_{self.operator}_{self.period_before}months"
            
    def _predictor_generator(self,var_name):
        """Produce one column from raw data"""
        self._separate_variables(var_name)
        self._check_period()
        var_name = self._rename_varname()
        
        var = self.raw_data_df
        #var['wym_str'] = var.wym.apply(lambda x: self.months_wy[x-1])
       
        var = self.raw_data_df[self.raw_data_df.wym <= self.month_initialisation_index]
        var = var[var.wym > self.month_initialisation_index - self.period_before]
        var = var.groupby("wy_simple").agg({self.variable: [self.operator]}).reset_index()
        var = var.rename({self.variable: var_name}, axis=1)
        var = var.set_index("wy_simple")
        var.columns = var.columns.droplevel(1)
        return var
    
    def _get_predictors(self):
        "Merge the different columns from raw data"
        X_list = [self._predictor_generator(var_name = predictor) for predictor in self.predictor_list]
        X = pd.concat(X_list, axis=1)
        X = X.dropna(axis=0)
        self.X = X
        
        
@dataclass            
class Predictant:
    """Agreggate flow into volume as the target variable """

    def __post_init__(self):
        self._flows_in_forecast_period()
        self._volume_from_flows()
        self._flows_as_predictants()
    
    def _flows_in_forecast_period(self) -> None:
        """Filter monthly flows between two months."""
        cond_start = self.monthly_flows.wym >= self.init_forecast_index +1
        cond_end = self.monthly_flows.wym <= self.end_forecast_index +1
        self.q_period_wy = self.monthly_flows.loc[(cond_start) & (cond_end)][["wy_simple", "wym", "Q_mm"]]
        
    def _volume_from_flows(self):
        """Transform monthly flows into volume"""
        y = self.q_period_wy.groupby(["wy_simple"]).sum()["Q_mm"].reset_index().set_index("wy_simple")
        self.y = y.rename(columns={'Q_mm':'volume_mm'})
        
    def _flows_as_predictants(self):    
        self.q = self.q_period_wy.pivot_table(values="Q_mm", index="wy_simple", columns="wym")
        self.q.columns = self.months_forecast_period
        self.q_months = list(self.q.columns)
        

@dataclass
class PreprocessedData(ForecastWindow,CatchmentData,Predictors,Predictant):
    """Initialised input data for the regression and classification model."""
    month_initialisation: str ="dic"
    window_strategy:str = 'dynamic'
    catchment_code: str = "5410002"
    wy_holdout: int = 2016
    

    def __post_init__(self):
        """Initiate methods."""
        ForecastWindow.__post_init__(self)
        CatchmentData.__post_init__(self)
        Predictors.__post_init__(self)
        Predictant.__post_init__(self)

        self._common_wateryears_X_q()
        self._set_train_data()
        self._set_test_data()
        

    def _common_wateryears_X_q(self) -> None:
        """Finds common years in predictors and flows, excludes hold-out year."""
        wys_X = pd.Series(self.X.index)
        wys_q = pd.Series(self.q.index)
        
        wys_common = pd.merge(wys_q,wys_X)
        wys_common = wys_common[wys_common.wy_simple != self.wy_holdout]
        wys_common = list(wys_common.wy_simple)
        
        self.wy_train = wys_common
        self.wy_init = min(self.wy_train)
        self.wy_end = max(self.wy_train) 
        
    def _set_train_data(self) -> None:
        """Set train data based on available data in X and q."""
        # training data: leave hold-out year out
        self.X_train = self.X.drop(self.wy_holdout, errors = 'ignore').loc[self.wy_train]
        self.y_train = self.y.drop(self.wy_holdout, errors = 'ignore').loc[self.wy_train].values.ravel()
        self.q_train = self.q.drop(self.wy_holdout, errors = 'ignore').loc[self.wy_train]
                
    def _set_test_data(self) -> None:
        """Set test data based on holdout year."""
        if self.wy_holdout in self.X.index:
            self.X_test = self.X.loc[[self.wy_holdout]]
        else:
            raise ValueError("You need predictors for forecast period (see wy_holdout).")
            
        if self.wy_holdout in self.q.index:                
            self.q_test = self.q.loc[[self.wy_holdout]]
            self.y_test = self.y.loc[[self.wy_holdout]]
   
if __name__ == '__main__':
    data = PreprocessedData(
        month_initialisation = "may",
        catchment_code= "5410002",
        predictor_list=["pr_sum_-1months","tem_mean_3months"],
        wy_holdout=2019
        )

    y_train = data.y_train
    q_train = data.q_train
    print(q_train.iloc[0])
    print(y_train[0])
