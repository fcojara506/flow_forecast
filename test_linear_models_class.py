# modified from https://machinelearningmastery.com/robust-regression-for-machine-learning-in-python/

# compare robust regression algorithms on a regression dataset with outliers

from class_regression import RegressionModel
# Regression models
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from functools import reduce 
# pandas
import pandas as pd
# charts
import matplotlib.pyplot as plt
import seaborn as sns

plt.close("all")  # close all plots


def get_models():
    """
    It creates a dictionary of models, where the keys are the names of the models and the values are the
    models themselves
    :return: A dictionary of models
    """
    models = dict()  # create dictionary of models
    models['Linear'] = linear_model.LinearRegression()
    models['Ridge'] = linear_model.Ridge(alpha=0.6)
    models['Lasso'] = linear_model.Lasso(alpha=0.6)
    models['ElasticNet'] = linear_model.ElasticNet(l1_ratio=0.9)
    models['BayesianRidge'] = linear_model.BayesianRidge()
    models['PLS'] = PLSRegression(n_components=1)
    models['Huber'] = linear_model.HuberRegressor()
    models['RANSAC'] = linear_model.RANSACRegressor(random_state=42)
    models['TheilSen'] = linear_model.TheilSenRegressor(random_state=42)
    models['poli3'] = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge(alpha=1e-3))
    #models['spline'] = make_pipeline(SplineTransformer(n_knots = 10, degree=3), linear_model.Ridge(alpha=1e-3))
    return models


# get the cross-validation scores and predictions

def cv_regression_models(X_train, y_train, model_list, cv):
    """
    The function takes in the training data, the model list, and cross-validation type.
    It then returns the cross-validation scores, the predictions, and the model list.
    
    :param X_train: the training data
    :param y_train: the target variable
    :param model_list: a dictionary of models to be evaluated
    :param cv: number of folds
    :return: The rmse, y_pred, and model_list
    """
    rmse, y_pred, pbias = dict(), dict(), dict()

    # scale data to plot
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)

    for name, model in model_list.items():
        #scale model
        model = make_pipeline(scaler,model)
        # evaluate the model
        y_pred[name], rmse[name], pbias[name] = dict(),dict(),dict()#evaluate_model_cv(X_train, y_train, model, name, cv, True)

    y_pred = pd.concat(y_pred, names=["model"]).reset_index()
    y_pred = pd.merge(y_pred, X_train_scaled, on='wy_simple').melt(id_vars=["model", "wy_simple", "y_pred"])

    return pd.concat(rmse), y_pred, pd.concat(pbias), X_train_scaled

def run_models_cv(month_init,predictors):
    predictors_str = ','.join(predictors)
    # load the training/test dataset
    data = RegressionModel(month_initialisation = month_init,
                             catchment_code= "5410002",
                             predictor_list=predictors)
    
    
    model_list = get_models()
    #y_cv, rmse_cv , pbias_cv = dict(),dict(),dict()
    df = dict()
    for name, model in model_list.items():
        
        data.set_model_parameters(model_name=name,
                                  model_scaler=StandardScaler(),
                                  model_regressor=model,
                                  cv = LeaveOneOut())
        alias = (name,predictors_str)
        data_frames = data.evaluate_model_cv(
            data.X_train, data.y_train, data.model, data.model_name, cv=data.cv)
        
        
        df[alias]   = reduce(lambda  left,right: pd.merge(left,right,on=['wy_simple']), data_frames)

    df = pd.concat(df, names=["model","pred"]).reset_index()
    
    catchment_code = data.catchment_code
    volume_span_text = data.volume_span_text
    

    X_train = data.X_train
    y_train = pd.DataFrame(data.y_train,index=X_train.index,columns=["y_train"])
    
    return df,X_train,y_train,catchment_code,volume_span_text


def plot_metric_df(metric,metric_df,catchment_code,volume_span_text,month_init,export=False):
    
    fig=plt.figure(num=1, figsize=(7, 7), dpi=500)
    
    sns.boxplot(x=metric,
                y='model',
                hue='pred',
                data=metric_df,
                palette="Set2",
                showmeans=True,
            meanprops={"marker": "x", "markerfacecolor": "white", "markeredgecolor": "red"})
    
    if metric == 'pbias': plt.axvline()
    
    
    plt.legend(loc='upper center',
               bbox_to_anchor=(0.5, -0.1),
               frameon=False,
               title="Predictors")
    
    plt.title(f"{metric.upper()} para varios modelos de regresión. Initializado 1 {month_init}")
    plt.ylabel('Modelo Regresión')
    plt.xlabel(f'{metric.upper()} {volume_span_text} (mm)')
    plt.show()
    
    filename_fig = f"data_output/pronostico_volumen/Figures/metrics/{catchment_code}/{metric.upper()}_predictors-models_test_{month_init}_{catchment_code}.png"
    if export:
        plt.savefig(filename_fig, bbox_inches='tight')
   
def plot_scatter(df_pred,df_train,catchment_code,volume_span_text,export=False):
    ## figure 3: Predictor vs Prediction for each model
    fig = plt.figure(num=3, figsize=(9, 6), dpi=500)
    sns.lineplot(x="pr_acum", y="y_pred", hue="model", data=df_pred)
    
    sns.scatterplot(x="pr_acum", y="y_train", data=df_train)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),frameon=False, title="Model Regresión")
    plt.title('Predicción (volumen) vs Predictor. Initializado 1 {month_init}')
    plt.xlabel('Precipitación acumulada')  # x-axis label
    plt.ylabel(f'Volumen  {volume_span_text}  (mm)')  # 'Cumulative flow'
    plt.show()
    
    if export:
        filename_fig = f"data_output/pronostico_volumen/Figures/metrics/{catchment_code}/Scatter_Xtrain_ypred_predictors-models_test_{month_init}_prAcum_{catchment_code}.png"
        plt.savefig(filename_fig, bbox_inches='tight')
    plt.close("all")  # close all plots
    
def test_predictors_months(month_init = 'may'):
    predictors_list = [["pr_acum", "tem_avg_3mons"],["pr_acum"]]  # list of predictors
       
    df = dict()
    X_train = dict()
    y_train = dict()
    
    for predictors in predictors_list:
        predictors_str = ','.join(predictors)
        alias = (predictors_str)
        df[alias], X_train[alias],y_train[alias] ,catchment_code,volume_span_text = run_models_cv(month_init,predictors)
            
    df = pd.concat(df,names=["pred"]).reset_index(drop=True)
    ## figure 1: RMS error in cross validation for the different models    
    plot_metric_df("rmse", df,catchment_code,volume_span_text,month_init, export = False)
    ## figure 2: Percentage bias in cross validation for the different models
    plot_metric_df("pbias", df,catchment_code,volume_span_text,month_init,export = False)
    ## figure 3: Predictor vs Prediction for each model
    X_train = pd.concat(X_train,names=["pred"]).reset_index()[["wy_simple","pr_acum"]].drop_duplicates()
    y_train = pd.concat(y_train,names=["pred"]).reset_index()[["wy_simple","y_train"]].drop_duplicates()
    df_train = pd.merge(X_train,y_train)
    df_pred = pd.merge(df[df.pred == "pr_acum"],X_train)
    
    plot_scatter(df_pred,df_train,catchment_code,volume_span_text,export=False)
    
    return 

month_init='may'
test_predictors_months(month_init = month_init)
      



