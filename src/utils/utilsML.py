from .libreries import *
from .utilsEDA import *

            ###
# **Funciones Machine Learning:**
            ###

# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Objetivo: 
    ---
    Generate a simple plot of the test and traning learning curve.

    args.
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
        
    x1 = np.linspace(0, 10, 8, endpoint=True) produces
        8 evenly spaced points in the range 0 to 10
    """

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def eval_metrics(y_pred,y_test,clf=True):
    '''
    Objetivo: 
    ---
    Evaluar el modelo con las métricas que correspondan.

    args.
    ---
    y_pred: la predicción realizada por el modelo. 
    y_test: el resultado real del test. 
    clf: bool; True: si es clasificación.
               False: si es regresión.

    ret.
    ---
    dict; resultado de las métricas.

    '''

    if clf:
        #confusion_mtx = confusion_matrix(y_test,y_pred)
        clf_metrics = {
            'ACC' : accuracy_score(y_test,y_pred),
            'Precision' : precision_score(y_test,y_pred),
            'Recall' : recall_score(y_test,y_pred),
            'F1' : f1_score(y_test,y_pred),
            'ROC' : roc_auc_score(y_test,y_pred),
            'Jaccard' : jaccard_score(y_test,y_pred)
        }

        #print(pd.DataFrame({'Values':clf_metrics.values()},index=clf_metrics.keys()))

        return clf_metrics

    else:

        reg_metrics = {
            'MAE' : mean_absolute_error(y_test,y_pred),
            'MAPE' : mean_absolute_percentage_error(y_test,y_pred),
            'MSE' : mean_squared_error(y_test,y_pred),
            'R2' : r2_score(y_test,y_pred)
        }   

        #print(pd.DataFrame({'Values':reg_metrics.values()},index=reg_metrics.keys()))

        return reg_metrics  


def baseline(data, target, base_model = None, clf = True, file_name = 'metrics.csv', dir_file = 'model/model_metrics', tsize = 0.2, random = 77):
    '''
    Objetivo: 
    ---
    Crear un modelo inicial orientativo.

    *args.
    ----
    data: pd.DataFrame; el dataset completo, con los valores numéricos.
    target: str; nombre de la columna objetivo, variable dependiente.
    base_model: estimador que se va a utilizar. Predeterminadamente se utilizar RandomForest(). (opcional)
    clf: True/False; si es un dataset de clasificación (True) si es de regresión (False). (opcional)
    tsize: float; tamaño del test [0.0,1.0]. (opcional)
    random: int; random state, semilla. (opcional)

    *ret.
    ----
    metricas de evaluación del modelo y el pack: 
        model_pack = {

            'trained_model' : estimator,
            'Xytest' : [X_test, y_test],
            'Xytrain' : [X_train, y_train],
            'ypred' : y_pred
        }

    '''

    if base_model == None:
        if clf:
            base_model = RandomForestClassifier()
        else:
            base_model = RandomForestRegressor()

    X = data.drop([target], axis=1)
    y = data[target].copy()
    #print('X shape: ',X.shape, '; y shape: ',y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = tsize, random_state = random)

    estimator=base_model.fit(X_train,y_train)
    y_pred=estimator.predict(X_test)
    model_pack = {

        'trained_model' : estimator,
        'Xytest' : [X_test, y_test],
        'Xytrain' : [X_train, y_train],
        'ypred' : y_pred
    }

    metrics = eval_metrics(y_pred,y_test,clf)
    
    dict4save(metrics, file_name, dir_file, addcols=True, cols='model', vals=str(base_model),sep=';')
    
    return metrics, model_pack

def choose_params(model,clf = True):
    '''
    Objetivo: 
    ---
    Elegir los parametros a probar para un modelo concreto.

    *args.
    ----
    model: modelo del cual se quieren los parametros.
    clf: bool; True: si se trata de un modelo de clasificación. 

    *ret.
    ----
    dict; con los parametros a probar.

    '''
    if clf :

        clf_params = {

            'LogReg' : {

                'penalty' : ['l1','l2','elasticnet','none'],
                'class_weight' : ['none','balanced'],
                'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter' : [100,150,200,500]
            },

            'KNNC' : {

                'n_neighbors' : [3,5,7,9,11,13,15],
                'weights' : ['uniform','distance'],
                'algorithm' : ['ball_tree','kd_tree','brute','auto'],
                'leaf_size' : [20,30,40],
                'p' : [1,2]

            },

            'DTC' : {
                
                'criterion' : ['log_loss','gini','entropy'],
                'splitter' : ['best','random'],
                'max_depth' : []

            },
            'ETC' : {},
            'RFC' : {},
            'BagC' : {},
            'AdaBC' : {},
            'GBC' : {},
            'SVC' : {},
            'XGBC' : {},
            'VC' : {}
        }

        return clf_params[model]

    else :

        reg_params = {

            'LinReg' : {},
            'KNNR' : {},
            'GNBR' : {},
            'BNBR' : {},
            'ENR' : {},
            'DTR' : {},
            'ETR' : {},
            'RFR' : {},
            'BagR' : {},
            'AdaBR' : {},
            'GBR' : {},
            'SVR' : {},
            'XGBR' : {}
        }

        return reg_params[model]

def choose_models(model, params, clf = True):
    '''
    Objetivo: 
    ---
    Elegir el modelo o los modelos que correspondan.

    *args.
    ----
    model: str; modelo que se quiere seleccionar. 
        'all': selecciona todos los modelos. 

    *ret.
    ----
    El/los modelos seleccionados.

    '''
    
    if clf :
        if params == None:

            classification_models={

                'LogReg' : LogisticRegression(),
                'KNNC' : KNeighborsClassifier(),
                'DTC' : DecisionTreeClassifier(),
                'ETC' : ExtraTreeClassifier(),
                'RFC' : RandomForestClassifier(),
                'BagC' : BaggingClassifier(), 
                'AdaBC' : AdaBoostClassifier(),
                'GBC' : GradientBoostingClassifier(),
                'SVC' : SVC(),
                'XGBC' : XGBClassifier(),
                'VC': VotingClassifier(estimators=[('RFC',RandomForestClassifier())]),
                'LDA': LinearDiscriminantAnalysis()
            }

        else:
            classification_models={

                'LogReg' : LogisticRegression(params),
                'KNNC' : KNeighborsClassifier(params),
                'DTC' : DecisionTreeClassifier(params),
                'ETC' : ExtraTreeClassifier(params),
                'RFC' : RandomForestClassifier(params),
                'BagC' : BaggingClassifier(params), 
                'AdaBC' : AdaBoostClassifier(params),
                'GBC' : GradientBoostingClassifier(params),
                'SVC' : SVC(params),
                'XGBC' : XGBClassifier(params),
                'VC': VotingClassifier(params),
                'LDA': LinearDiscriminantAnalysis(params)
            }


        if model == 'all' and params == None:
            return classification_models

        else:
            return classification_models[model]

    else : 

        if params == None:

            regression_models={

                'LinReg' : LinearRegression(),
                'KNNR' : KNeighborsRegressor(),
                'GNBR' : GaussianNB(),
                'BNBR' : BernoulliNB(),
                'ENR' : ElasticNet(),
                'DTR' : DecisionTreeRegressor(),
                'ETR' : ExtraTreeRegressor(),
                'RFR' : RandomForestRegressor(),
                'BagR' : BaggingRegressor(), 
                'AdaBR' : AdaBoostRegressor(),
                'GBR' : GradientBoostingRegressor(),
                'SVR' : SVR(),
                'XGBR' : XGBRegressor()
                
            }

        else:

            regression_models={

                'LinReg' : LinearRegression(params),
                'KNNR' : KNeighborsRegressor(params),
                'GNBR' : GaussianNB(params),
                'BNBR' : BernoulliNB(params),
                'ENR' : ElasticNet(params),
                'DTR' : DecisionTreeRegressor(params),
                'ETR' : ExtraTreeRegressor(params),
                'RFR' : RandomForestRegressor(params),
                'BagR' : BaggingRegressor(params), 
                'AdaBR' : AdaBoostRegressor(params),
                'GBR' : GradientBoostingRegressor(params),
                'SVR' : SVR(params),
                'XGBR' : XGBRegressor(params)
                
            }

        if model == 'all'and params == None:
            return regression_models

        else:
            return regression_models[model]

def save_model():
    
    return 