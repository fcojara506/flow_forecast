from celluloid import Camera
#import data
from class_preprocess_data import PreprocessedData
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# export data
import numpy as np
import matplotlib.pyplot as plt  
#import matplotlib.animation as animation

month_initialisation  = "may"
data = PreprocessedData(month_initialisation = month_initialisation,
                         catchment_code= "5410002",
                         predictor_list=["pr_acum"],
                         wy_holdout=2016,
                         wy_init=1981,
                         wy_end=2019)
# data in environment
X, y, X_holdout,wy_data = np.array(data.X_train),data.y_train,data.X_test, np.array(data.wy_train)


# Select one regression model and its parameters
model_scaler = StandardScaler()
model_regressor = HuberRegressor()
model = make_pipeline(model_scaler,model_regressor)
# cross validation (this case LOO-CV)
cv = LeaveOneOut()

def animate(iterator):
    
    # split the data manually into training and test with LOOCV
    train_index, test_index = list(cv.split(X))[iterator]
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    wy_train,wy_test = wy_data[train_index],wy_data[test_index]
    
    # fit and predict using the regression model
    model.fit(X_train,y_train)
    
    # get some important stats
    y_train_avg = np.mean(y_train)
    
    # prediction
    y_pred = model.predict(X_test)
    # rmse
    rmse = mean_squared_error(y_true=y_test,y_pred=y_pred, squared= False)
    rmse = round(rmse,1)
    # bias
    pbias = np.sum(y_pred-y_test)/np.sum(y_test)*100
    pbias = round(pbias,1)
    # r2 from the training
    r2 = r2_score(y_true=y_train,y_pred= model.predict(X_train))
    r2 = round(r2,2)
    
    # title
    ax.text(0.01, 1.14, "Regresión Volumen vs Precipitación en Validación Cruzada ", transform=ax.transAxes, fontsize=13)
    ax.text(0.01, 1.08, "Inicializado 1 de " + month_initialisation, transform=ax.transAxes, fontsize=13)
    ax.text(0.01, 1.02, "Año Hidrológico " + str(*wy_test), transform=ax.transAxes, fontsize=13)
    
    # plots
    p1, = ax.plot(X_train,y_train,'o',label = "Datos entrenamiento", color = "black")
    p2, = ax.plot(X_train,model.predict(X_train),label = "Modelo Regresión (Entrenamiento)",color = "blue")
    p3  = ax.axhline(y=y_train_avg,label = "Volumen promedio (Entrenamiento)", color = "grey")
    p4, = ax.plot(X_test,y_test,'x',label = "Dato apartado",markersize=10,markeredgewidth = 2, color = "red")
    p5, = ax.plot(X_test,y_pred,'p',label = "Predicción del dato apartado", markersize=5,markeredgewidth = 2, color = "lawngreen")
    
    # metrics on screen
    ax.text(1.1, 0.2, "Métricas en validación" , wrap=True, horizontalalignment='left',fontsize=12, transform=ax.transAxes)
    ax.text(1.1, 0.15, f"RMSE : {rmse} mm" , wrap=True, horizontalalignment='left',fontsize=11, transform=ax.transAxes)
    ax.text(1.1, 0.1, f"Perc. Sesgo : {pbias}%" , wrap=True, horizontalalignment='left',fontsize=11, transform=ax.transAxes)
    ax.text(1.1, 0.05, f"Coef. r2 (Entrenamiento) : {r2} " , wrap=True, horizontalalignment='left',fontsize=11, transform=ax.transAxes)
   
    # labels
    plt.xlabel("Precipitación acumulada a la fecha desde marzo (mm)")
    plt.ylabel(f"Volumen {data.volume_span_text} (mm)")
    plt.ylim(0,1000)
    plt.xlim(0,max(X)*1.1)
    ax.legend(handles=[p1,p2,p3,p4,p5],loc='upper left', frameon=False,bbox_to_anchor=(1.05, 0.9))

    
#create loop animation
fig, ax = plt.subplots(1,figsize=(8,5), dpi=500)
camera = Camera(fig)
for i,x in enumerate(X):
    animate(i)
    fig.tight_layout()
    camera.snap()
 
animation = camera.animate(repeat = False, interval = 2000, repeat_delay = 1000 )
animation.save(f'animation_LOOCV_{month_initialisation}.gif',dpi = 500)

