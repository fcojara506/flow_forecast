from functions_preprocess_data import months_initialisation, split_train_test_data
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot


# retrieve the model to be evaluate
def scale_model(model):
    # Create the pipeline: pipeline
    pipeline = make_pipeline(StandardScaler(), model)
    return pipeline

# evaluate the model using a given test condition
def evaluate_model(cv):
    # get the model
    model = scale_model(Ridge())
    # evaluate the model
    scores = cross_val_score(
        model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)*-1
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)

    # return scores
    return scores.mean(), scores.min(), scores.max()


X_train, y_train, X_test, volume_span_text = split_train_test_data(
    month_initialisation="jun")
# calculate the ideal test condition
ideal_mean, ideal_min, ideal_max = evaluate_model(cv=LeaveOneOut())

#print('Ideal: %.3f' % ideal_avg)
print('LOOCV, accuracy=%.3f (%.3f,%.3f)' % (ideal_mean, ideal_min, ideal_max))
# define folds to test
folds = range(2, len(X_train)-1)
# record mean and min/max of each set of results
means, mins, maxs = list(), list(), list()
# evaluate each k value
for k in folds:
    # define the test condition
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    # evaluate k value
    k_mean, k_min, k_max = evaluate_model(cv)
    # report performance
    print('> folds=%d, accuracy=%.3f (%.3f,%.3f)' % (k, k_mean, k_min, k_max))
    # store mean accuracy
    means.append(k_mean)
    # store min and max relative to the mean
    mins.append(k_mean - k_min)
    maxs.append(k_max - k_mean)
# line plot of k mean values with min/max error bars
pyplot.errorbar(folds, means, yerr=[mins, maxs], fmt='o')
# plot the ideal case in a separate color
pyplot.plot(folds, [ideal_mean for _ in range(len(folds))], color='r')
# show the plot
pyplot.show()
