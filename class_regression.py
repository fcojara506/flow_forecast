
from class_preprocess_data import PreprocessedData
from class_knn_flows import KnnModel
from matplotlib.transforms import blended_transform_factory
from functions_metrics import deterministic_scores, ensemble_scores, pbias
from sklearn.model_selection import cross_val_predict, LeaveOneOut, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import HuberRegressor
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)

class RegressionModel(PreprocessedData,KnnModel):
    """Define Regression Model to Forecast volume."""
    
        
    def get_data(self):
        """Load data from PreprocessedData Class."""
        super().__post_init__()
  
    def set_model_parameters(self, 
                             model_name="Huber",
                             model_scaler = StandardScaler(),
                             model_regressor= HuberRegressor(),
                             cv = LeaveOneOut()):
        """Select model and parameters."""
        self.model_name = model_name
        self.model_scaler = model_scaler
        self.model = make_pipeline(model_scaler,model_regressor)
        self.cv = cv
        
    @staticmethod    
    def evaluate_model_cv(X, y, model, cv, rownames=True):

        # evaluate regression in cross-validation
        rmse = cross_validate(
            model,
            X,
            y,
            scoring='neg_root_mean_squared_error',
            cv=cv,
            n_jobs=-1,
            return_train_score=True)["train_score"]
        rmse = np.absolute(rmse)  

        pbias_score = cross_validate(
            model,
            X,
            y,
            scoring=make_scorer(pbias, greater_is_better=False),
            cv=cv,
            n_jobs=-1,
            return_train_score=True)["train_score"]

        #obtain the prediction in cross-validation. n_jobs = -1 (parallel)
        y_pred_cv = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            n_jobs=-1)

        # from numpy array to DataFrame with water year in row index 
        if rownames:
            wy = X.index.astype(int)
            rmse = pd.DataFrame(rmse).set_index(wy).rename(columns={0: 'rmse'})
            pbias_score = pd.DataFrame(pbias_score).set_index(
                wy).rename(columns={0: 'pbias'})
            
            y_pred_cv = pd.DataFrame(y_pred_cv).set_index(
                wy).rename(columns={0: 'y_pred'})

        return y_pred_cv, rmse, pbias_score
    
    @staticmethod
    def ensemble_generator(y, rmse, n_members):
        """
        It takes a vector of predicted volumes and a vector of RMSEs and returns a matrix of random samples
        from a normal distribution with mean = predicted volume and std = RMSE
        
        :param y: the predicted volume
        :param rmse: the root mean squared error of the model
        :param n_members: the number of members in the ensemble, defaults to 1000 (optional)
        :return: A dataframe with the same number of columns as the number of years in the forecast period.
        """
        if len(y) > 1:
            # initialise a list of volumes
            ensemble_volume = list()
        # generate a random vector for each year with avg = predicted_volume and std = rmse from the cross-validation
            for wy in y.index:
                # define a random normal distribution based on the deterministic prediction
                # the scale based on the RMSE of each regression
                centre = y.loc[wy]
                variation = rmse.loc[wy]
                # random sample
                ensemble_vol_wy = np.random.normal(
                    loc=centre, scale=variation, size=n_members)
                # transform to DataFrame
                ensemble_vol_wy = pd.DataFrame(ensemble_vol_wy)
                # append column to DataFrame
                ensemble_volume.append(ensemble_vol_wy)

            ensemble_volume = pd.concat(ensemble_volume, axis=1)
            #ensemble_volume.columns = y.index

        else:
            ensemble_volume = np.random.normal(
                loc=y[0], scale=rmse, size=n_members)

        ensemble_volume = pd.DataFrame(ensemble_volume)
        ensemble_volume.columns = y.index

        return ensemble_volume
    
    def forecast_vol_ensemble(self,n_ens_members):

        # evaluate the model in cross-validation (Leave one out)
        self.y_cv, self.rmse_cv, self.pbias = self.evaluate_model_cv(
            self.X_train, self.y_train, self.model, cv=self.cv)

        # fit model
        self.model.fit(self.X_train, self.y_train)

        # predict deterministic hold-out data
        y_fore = self.model.predict(self.X_test)
        self.y_fore = pd.DataFrame(y_fore, index=list(self.X_test.index))

        # predict training data to compute scores
        y_pred_train = self.model.predict(self.X_train)
        # compute the RMSE of the training
        rmse_model = mse(self.y_train, y_pred_train, squared=False)

        ## generate ensembles from predicted volumes
        y_ens_cv = self.ensemble_generator(
            y=self.y_cv, rmse=self.rmse_cv, n_members=n_ens_members)
        
        y_ens_fore = self.ensemble_generator(
            y=self.y_fore, rmse=rmse_model, n_members=n_ens_members)

        # deterministic and probabilistic metrics
        y_ens_avg = y_ens_cv.mean()  # ensemble mean
        # format ensemble to use in functions
        y_ens = np.array(y_ens_cv).transpose()
        y_train = np.array(self.y_train)

        uni_scores = deterministic_scores(y_train, y_ens_avg)
        ens_scores = ensemble_scores(y_train, y_ens)
        self.score_dict = uni_scores | ens_scores
        
        self.y_ens_cv = y_ens_cv
        self.y_ens_fore = y_ens_fore
        self.n_ens_members = n_ens_members
    @staticmethod  
    def wy_to_year(wy,wym) -> int:
        """Take a wateryear(wy) to gregorian year"""
        gregorian_year = wy if wym<10 else wy+1
        return gregorian_year
        
    def input_plot_hindcast(self,y_ens_cv, y_ens_fore,y_train):
        
        # quantiles observations and ensemble forecast of hold-out year
        quantiles_obs = np.quantile(y_train, [0.05, 0.5, 0.95])
        quantiles_fore = np.quantile(y_ens_fore, [0.25, 0.5, 0.75])
        
        # compute uncertainty error as max between the median and the interquantile limits
        error_range_fore = (quantiles_fore - np.median(y_ens_fore))
        error_range_fore = max(abs(error_range_fore[error_range_fore != 0]))
        
        # median and range into string
        text_forecast_range = '{0:.1f}'.format(
            np.median(y_ens_fore)) + ' ± ' + '{0:.1f}'.format(error_range_fore) + ' mm'

        # create dataframe to insert data in charts
        # water years dataframe
        wy_train = pd.DataFrame(self.wy_train, columns=["wy_simple"])
        y_ens = pd.concat([y_ens_cv, y_ens_fore], axis=1)
        y_ens = y_ens[sorted(y_ens.columns.to_list())]

        # observations (training data) data frame
        df_train = pd.concat([wy_train, pd.DataFrame(
            y_train, columns=["value"])], axis=1, names=["wy_simple", "value"])
        df_train = df_train.append(
            {'wy_simple': self.wy_holdout, 'value': None}, ignore_index=True)
        df_train = df_train.sort_values(by='wy_simple')

        # add new data to metadata
        self.quantiles_avg_obs = quantiles_obs
        self.text_forecast_range = text_forecast_range
        self._set_label_texts()

        return df_train, y_ens

    def _set_label_texts(self):    
        # useful text for charts
        year_init = self.wy_to_year(self.wy_holdout,self.pred_init_month)
        year_end =  self.wy_to_year(self.wy_holdout,self.pred_end_month)
        
        year_initialisation = self.wy_to_year(self.wy_holdout, self.month_index+1)
        self.date_initialisation = f"1 {self.month_initialisation} {year_initialisation}"
        self.volume_span_text_v2 = f"[{self.months_wy[self.pred_init_month - 1]}/{year_init},{self.months_wy[self.pred_end_month - 1]}/{year_end}]"
        self.volume_span_text = f"[{self.months_wy[self.pred_init_month - 1]},{self.months_wy[self.pred_end_month - 1]}]"
        
    def plot_forecast_boxplot_obs_points(self,y_ens, df_train, xlabel, ax):
        ### chart 1 with forecast in chronological order
        df_train.wy_simple = df_train.wy_simple.astype(int).astype(str)

        # boxplot
        sns.violinplot(data=y_ens, color='skyblue',
                       ax=ax, linewidth=0.5)
        # observations as points x
        p = sns.scatterplot(x="wy_simple", y="value", data=df_train,
                            marker='x', color='red', s=100, label='Observaciones', ax=ax)

        # chart setup
        ax.tick_params(labelrotation=90)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Volumen (mm)")
        ax.legend(loc='upper right', title="")
        ax.set_ylim(bottom=0)

        # color xlabel for the hold-out year
        for tick_label in p.axes.get_xticklabels():
            if tick_label.get_text() == str(self.wy_holdout):
                tick_label.set_color("dodgerblue")

        # text with 5%, 50% and 95% percentile of training observations
        tform = blended_transform_factory(ax.transAxes, ax.transData)

        for quantile, yc in zip(['5%', '50%', '95%'], self.quantiles_avg_obs):
            ax.axhline(y=yc, zorder=1, linestyle='--',
                       linewidth=1.5)  # horizontal lines
            ax.annotate(quantile + " (" + str(round(yc)) + ")",
                        xy=(1.06, yc), xycoords=tform, ha='center', va='center')

        return ax

    def subplot_boxplot_chronological_median_order(self, export=False, subplot=True):
        # formatted input data
        df_train, y_ens = self.input_plot_hindcast(self.y_ens_cv, self.y_ens_fore, self.y_train)
        
        if subplot:
            # initialise subplots
            fig, axs = plt.subplots(2, figsize=(7, 4))
            plt.suptitle(self.gauge_name)
            axs[0].set_title(f"Pronóstico de volumen {self.volume_span_text}. Inicializado {self.date_initialisation}")

            #### chart 1 with forecast in chronological order
            xlabel = "Año hidrológico (orden cronológico)"
            
            self.plot_forecast_boxplot_obs_points(
                y_ens, df_train, xlabel, ax=axs[0])

            #### chart 2 with forecast using median to order xlabel
            meds = y_ens.median()
            meds.sort_values(ascending=False, inplace=True)
            y_ens = y_ens[meds.index]

            # transform or re-organise data
            df_train.wy_simple = df_train.wy_simple.astype(int)
            df_train = df_train.set_index("wy_simple", drop=False)
            df_train = df_train.reindex(meds.index)

            ## plot boxplot and obs in new order
            xlabel = "Año hidrológico (orden por mediana del pronóstico)"
            
            self.plot_forecast_boxplot_obs_points(
                y_ens, df_train, xlabel, ax=axs[1])
        else:
            # initialise subplots
            fig, axs = plt.subplots(1, figsize=(7.5, 4))
            plt.suptitle(self.gauge_name)
            axs.set_title(f"Pronóstico de volumen {self.volume_span_text}. Inicializado {self.date_initialisation}")

            #### chart 1 with forecast in chronological order
            xlabel = "Año hidrológico (orden cronológico)"
            
            self.plot_forecast_boxplot_obs_points(
                y_ens, df_train, xlabel, ax=axs)

        ## add more information to the chart in caption
        plt.figtext(0.5, -0.05, f"Pronóstico de volumen (mediana ± rango intercuartil/2) {self.volume_span_text_v2}: {self.text_forecast_range}", wrap=True, horizontalalignment='center', fontsize=11)

        # handle white spaces in the figure
        fig.tight_layout()

        # boolean to export
        if export:
            # figure output filename
            folder_output = f"data_output/pronostico_volumen/Figures/ensemble_forecast/{self.catchment_code}/"
            # create folder if it does not exist
            Path(folder_output).mkdir(parents=True, exist_ok=True)

            self.figure_vol_output = folder_output + \
                "EnsembleVolumeHindcast_" + \
                self.catchment_code  + "_" + \
                "1st"+self.month_initialisation+"_" + \
                "_AND_".join(self.predictor_list)+"_" + \
                self.volume_span_text + str(self.wy_holdout) + \
                ".png"
                
            plt.savefig(self.figure_vol_output, bbox_inches='tight')
        
