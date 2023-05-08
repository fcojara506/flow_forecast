# import machine learning knn functions
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.pipeline import make_pipeline
# metrics
from functions_metrics import deterministic_scores, ensemble_scores, percentile
from pathlib import Path
#charts
import matplotlib.pyplot as plt
import seaborn as sns
# classic data manupulation
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class KnnModel:
    """ Select nearest year's predictors based on knn to estimate flow divided by volume """
    
    def set_knn_parameters(self,*args,**kwargs):
        """ Set knn parameters"""
        knn_model = KNeighborsRegressor(*args,**kwargs)
    
        # save model's metadata
        self.knn_model = knn_model
        self.info['n_knn_neighbors']:self.knn_model.n_neighbors
    
    def scale_knn_model(self):
        """ model pipeline"""
        scaler = MinMaxScaler()
        self.knn_pipeline = make_pipeline(scaler, self.knn_model)  # scaled model
        self.cv_knn = LeaveOneOut()
        
    @staticmethod
    def ensemble_generator_qq(f, y_ens):
        y_ens_i = dict()
        wys = y_ens.columns.tolist()
    
        for wy in wys:
            y_i = y_ens[[wy]]
            f_i = f.loc[[wy]]
            y_ens_i[wy] = y_i.dot(f_i)
            
        return y_ens_i      
    
    def q_ensemble(self):
        # f:flow/volume
        f_train = self.q_train.div(self.q_train.sum(axis=1), axis=0)
        
        
        # evaluate the model in cross-validation (Leave one out)
        f_cv = cross_val_predict(self.knn_pipeline, self.X_train, f_train, cv=self.cv_knn, n_jobs=-1)
        f_cv = pd.DataFrame(f_cv, index=list(self.X_train.index), columns=f_train.columns)
    
        # fit model for train data
        self.knn_pipeline.fit(self.X_train, f_train)
    
        # predict hold-out data
        f_fore = self.knn_pipeline.predict(self.X_test)
        f_fore = pd.DataFrame(f_fore, index=list(self.X_test.index), columns=f_train.columns)
    
        ## generate ensembles from predicted f
        self.q_ens_cv = self.ensemble_generator_qq(f=f_cv, y_ens=self.y_ens_cv)
        self.q_ens_fore = self.ensemble_generator_qq(f=f_fore, y_ens=self.y_ens_fore)
        #self.compute_q_scores()
        
    def forecast_q_ensemble(self,*args,**kwargs):
        self.set_knn_parameters(*args,**kwargs)
        self.scale_knn_model()
        self.q_ensemble()
        self.compute_q_scores()
        
    def compute_q_scores(self):
        
        ### deterministic scores   
        
        # ensemble monthly average
        q_ens_cv_avg = pd.concat(self.q_ens_cv,names=["wy_simple", "ens"])
        q_ens_cv_avg = q_ens_cv_avg.groupby(by="wy_simple", axis=0).mean()
        q_ens_cv_avg = q_ens_cv_avg.melt(ignore_index = False, value_name = "q_avg").reset_index()
        
        # q_train in one column
        q_train_vector = self.q_train.melt(ignore_index = False, value_name = "q_train",var_name="month").reset_index()
        df_univar = pd.merge(q_ens_cv_avg,q_train_vector)
        
        uni_scores = deterministic_scores(df_univar.q_avg,df_univar.q_train)
        
        ### ensemble scores
        q_ens = pd.concat(self.q_ens_cv,names=["wy_simple", "ens"])
        q_ens = q_ens.melt(ignore_index=False,value_name = "q_ens",var_name="month").reset_index()
        q_ens = q_ens.pivot_table(values='q_ens',index=["wy_simple","month"], columns="ens")
        q_ens = q_ens.reset_index()
        
        q_ens_df = pd.merge(q_ens,q_train_vector)
        
        q_ens = np.array(q_ens_df.drop(["wy_simple","month","q_train"],axis=1))
        q_train_vector = np.array(q_ens_df["q_train"])
        
        ens_scores = ensemble_scores(q_train_vector,q_ens)
        scores_knn = uni_scores | ens_scores
        self.scores_knn = pd.DataFrame.from_dict(scores_knn,orient='index',columns = ['q']).T
        self.info['scores_q'] = self.scores_knn
        
    def plot_knn_flow(self,export: bool = False) -> None:
        #wym = self.months_forecast_period
        months_wy = pd.DataFrame(self.months_wy,columns=["wym"])
        
        # forecast
        df = self.q_ens_fore
        df = df[list(df)[0]]
        #df.columns = wym
        df = df.melt(var_name="wym")
        df = pd.merge(df,months_wy,how='right')
        
        # observations
        q_obs = self.monthly_flows.copy()[["wy_simple", "wym", "Q_mm"]]
        
        # observation water year hold.out (if exists)
        q = q_obs[q_obs.wy_simple == self.wy_holdout][["wym", "Q_mm"]]
        #q.wym = q.wym.apply(lambda x: self.months_wy[x-1])
        q['variable'] = 'wy_holdout'
        q = q.rename(columns={'Q_mm':'value'})
        
        # stats of other years
        q_obs_stats = q_obs[q_obs.wy_simple != self.wy_holdout][["wym", "Q_mm"]].reset_index(drop=True)
        q_obs_stats = q_obs_stats.groupby("wym").agg({"Q_mm": [np.mean, np.median, percentile(5), percentile(95)]})
        q_obs_stats.columns = q_obs_stats.columns.droplevel(0)
        q_obs_stats = q_obs_stats.reset_index()
        #q_obs_stats.wym = q_obs_stats.wym.apply(lambda x: self.months_wy[x-1])
        q_obs_stats = q_obs_stats.melt(["wym"])
        
        ## merge observation data
        q_merge = pd.merge(q,q_obs_stats,how='outer')
        q_merge.wym = q_merge.wym.apply(lambda x: self.months_wy[x-1])
        
        fig,ax = plt.subplots(1,figsize=(7, 4),dpi=400)
        # boxplot or violin of forecasts
        sns.violinplot(x=df.wym,y=df.value,color='skyblue')
        
        # lines of observations
        sns.lineplot(x=q_merge.wym,
                         y=q_merge.value,
                         hue=q_merge.variable,
                         #marker="",
                         color="red")
        
        # get handles and labels from the data so you can edit them
        h,l = ax.get_legend_handles_labels()
        
        
        # # keep same handles, edit labels with names of choice
        ax.legend(handles=h,
                   labels=[f'Target year ({self.wy_holdout})', 
                          f'Average [{self.wy_init},{self.wy_end}]',
                          f'Median [{self.wy_init},{self.wy_end}]',
                          f'Percentile 5% [{self.wy_init},{self.wy_end}]',
                          f'Percentile 95% [{self.wy_init},{self.wy_end}]',
                          ],
                   title = "Stream flow in guage station",
                  frameon=False)
        
        plt.ylim(bottom=0)
        
        plt.xlabel("Forecast horizon")
        plt.ylabel("Mean monthly flow (mm)")
        plt.title(f"Mean monthly flow. Initialised {self.date_initialisation}")
        plt.suptitle(self.gauge_name)
        
        # handle white spaces in the figure
        fig.tight_layout()
        # boolean to export
        if export:
            predictor_list="_AND_".join(self.predictor_list)
            # figure output filename
            folder_output = f"data_output/pronostico_caudal/Figures/ensemble_forecast/{self.catchment_code}/"
            # create folder if it does not exist
            Path(folder_output).mkdir(parents=True, exist_ok=True)
    
            self.figure_flow_output = f"{folder_output}flow_ensemble_forecast_{self.catchment_code}_1st{self.month_initialisation}_{predictor_list}_{self.volume_span_text}_{self.wy_holdout}.png"
                
            plt.savefig(self.figure_flow_output, bbox_inches='tight')
            
def main():
    KnnModel(num_neighbors=6)
    

if __name__ == '__main__':
    input_data = main()

