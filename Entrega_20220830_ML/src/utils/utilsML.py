from .libreries import *
from .utilsEDA import *
import warnings
warnings.filterwarnings('ignore')

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

    plt.figure(figsize=(15,8))
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
    model_str = str(base_model)[0:str(base_model).find('(')]

    dict4save(metrics, file_name, dir_file, addcols=True, cols='model', vals=model_str,sep=';')
    
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
                'max_iter' : [50,75,100,150,200]
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
                'max_depth' : [7,9,11,13,None],
                'max_features': ['log2','sqrt','auto'],
                'class_weight' : [None,'balanced']

            },

            'ETC' : {
                #'n_estimators': np.linspace(10,80,10).astype(int),
                'criterion': ['gini','entropy'],
                'max_depth' : [7,9,11,13,None],
                'max_features': ['log2','sqrt',None],
                'class_weight' : [None,'balanced'],
                'max_leaf_nodes' : [None,3,7,11]
            },

            'RFC' : {
                'n_estimators': np.linspace(10,150,10).astype(int),
                'criterion': ['gini','entropy'],
                'max_depth' : [7,9,11,13,None],
                'max_features': ['log2','sqrt',None],
                'class_weight' : [None,'balanced']
            },

            'BagC' : {
                #'base_estimator__class_weight': ['balanced'],
                #'base_estimator__criterion': ['gini'],
                #'base_estimator__max_depth': [7], 
                #'base_estimator__max_features': ['log2'], 
                #'base_estimator__splitter': ['best'],
                'n_estimators' : [10, 20, 30, 50, 100],
                'max_samples' : [0.05, 0.1, 0.2, 0.5]
            },
            'AdaBC' : {
                #'base_estimator__class_weight': ['balanced'],
                #'base_estimator__criterion': ['gini'],
                #'base_estimator__max_depth': [7], 
                #'base_estimator__max_features': ['log2'], 
                #'base_estimator__splitter': ['best'],
                'n_estimators' : [10, 20, 30, 50, 100]
            
            },

            'GBC' : [{
                #'base_estimator__class_weight': ['balanced'],
                #'base_estimator__criterion': ['gini'],
                #'base_estimator__max_depth': [7], 
                #'base_estimator__max_features': ['log2'], 
                #'base_estimator__splitter': ['best'],
                'n_estimators' : [10, 20, 30, 50, 100],
                'max_depth' : [7,9,11,13,None],
                'criterion': ['friedman_mse','mse'],
                'loss': ['log_loss','exponential']
            },
            {
              'loss' : ["log_loss"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01, 0.001],
              'max_depth': [4, 8,16],
              'min_samples_leaf': [100,150,250],
              'max_features': [0.3, 0.1]
              }
            ],

            'SVC' : [
                #{'C' : [1,10,50],'kernel' : ['poly','sigmoid','precomputed'],'degree' : [3,4],'class_weight' : [None,'balanced']},
                {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'class_weight' : [None,'balanced']},
                {'C': [1, 10, 100, 1000],  'kernel': ['rbf'],'class_weight' : [None,'balanced']}
            ],

            'XGBC' : {
                'nthread':[4], #when use hyperthread, xgboost may become slower
                'objective':['binary:logistic'],
                'learning_rate': [0.05], #so called `eta` value
                'max_depth': [4,5,6,7],
                'min_child_weight': [1, 5, 10, 11],
                'subsample': [0.6,0.8,1.0],
                'colsample_bytree': [0.6,0.7,1.0],
                'n_estimators': [5,50,100], #number of trees, change it to 1000 for better results
                'missing':[-999],
                'seed': [1337]
            },
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

def save_model(model,dirname):
    '''
    
    '''
    model_str = str(model)
    model_str = model_str[0:model_str.find('(')]
    ruta_dir = os.path.join(os.getcwd(), dirname)
    
    os.makedirs(ruta_dir,exist_ok=True)
    ruta_file = os.path.join(ruta_dir,f'{model_str}.pkl')
    
    
    if os.path.exists(ruta_file):
        for i in range(1,99):
            ruta_file = os.path.join(ruta_dir,f'{model_str}_{i}.pkl')
             
            if os.path.exists(ruta_file):
                x='otro intento'
            else:
                pickle.dump(model, open(ruta_file,'wb'))
                the_path = os.path.join(dirname,f'{model_str}_{i}.pkl')
                break
    else:
        pickle.dump(model, open(ruta_file,'wb'))
        the_path = os.path.join(dirname,f'{model_str}.pkl')

    print(f'Model {model_str} saved')
    
    return the_path 

def train_predict_best_model(data, target, model, params, scoring, tsize = 0.2, random = 77, scaling = False, balancing = False):
    '''
    
    '''
    # Separación data: 
    X = data.drop([target], axis=1)
    y = data[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = tsize, random_state = random)
    
    # Escalado:
    if scaling:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Balanceo
    if balancing:
        sm = SMOTEENN(random_state = random) 
        X_train, y_train = sm.fit_resample(X_train, y_train.ravel()) 

        

    # Entrenando al modelo: 
    estimator = GridSearchCV(model, params, scoring = scoring, refit = 'AUC', return_train_score = True)
    estimator.fit(X_train,y_train)

    # Predicción con el mejor estimador 
    y_pred=estimator.best_estimator_.predict(X_test)

    return estimator, X_test, y_test, X_train, y_train, y_pred

def save_all(model, estimator, params, metrics, file_name = 'metrics.csv', dir_file = 'model/model_metrics', dir_model_file = 'model'):
    '''
    Objetivo: 
    ---
    
    '''
    model_str = str(model)[0:str(model).find('(')]
    
    file2save = {'model':model_str,'params_tried': str(params),'best_params':str(estimator.best_params_)}
    file2save.update(metrics)
    
    # Guardar modelo:
    model_path = save_model(estimator.best_estimator_,dir_model_file)

    file2save.update({'model_path' : model_path})

    #Guardar archivo:
    dict4save(file2save, file_name, dir_file, addcols=False,sep=';')


def models_generator(data, target, model = None, params = None, clf = True, scaling = True, scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}, balancing = False, file_name = 'metrics.csv', dir_file = 'model/model_metrics', dir_model_file = 'model', tsize = 0.2, random = 77):
    '''
    Objetivo: 
    ---
    Crear un modelo inicial orientativo.

    args.
    ----
    data: pd.DataFrame; el dataset completo, con los valores numéricos.
    target: str; nombre de la columna objetivo, variable dependiente.
    base_model: estimador que se va a utilizar. Predeterminadamente se utilizar RandomForest(). (opcional)
    clf: True/False; si es un dataset de clasificación (True) si es de regresión (False). (opcional)
    tsize: float; tamaño del test [0.0,1.0]. (opcional)
    random: int; random state, semilla. (opcional)

    ret.
    ----
        model_pack = {

            'trained_model' : estimator,
            'Xytest' : [X_test, y_test],
            'Xytrain' : [X_train, y_train],
            'ypred' : y_pred
        }

    '''

    # Modelo por defecto:
    if model == None:
        if clf:
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()

    # Estimador entrenado y predicción: 
    estimator, X_test, y_test, X_train, y_train, y_pred = train_predict_best_model(data, target, model, params, scoring, tsize = 0.2, random = 77, scaling = scaling, balancing = balancing)

    # Evaluación de métricas:
    metrics = eval_metrics(y_pred,y_test,clf)
    
    # Guardar modelo y métricas obtenidas:
    save_all(model, estimator, params, metrics, file_name = file_name, dir_file = dir_file, dir_model_file = dir_model_file)

    # Variable de salida: 
    model_pack = {

        'trained_model' : estimator,
        'Xytest' : [X_test, y_test],
        'Xytrain' : [X_train, y_train],
        'ypred' : y_pred,
        'metrics' : metrics
    }

    return model_pack

