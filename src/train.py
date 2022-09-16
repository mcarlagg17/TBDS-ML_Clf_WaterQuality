from utils.libreries import *
from utils.utilsML import choose_models

#####################################################################
# -------------------------------------------------------------------
# Variables to select
# -------------------------------------------------------------------

model_of_ = 'c'         # c: classification models or 
                        # r: regression models (SUPERVISED MODELS)

encode = True           # True: if we need to encode or 
                        # False: if not 

unsupervised = False     # True: if is used unsupervised estimators 
                        # False: if not

supervised = True       # True: if is used supervised estimators 
                        # False: if not

deep_learning = False   # True: if is used deep learning 
                        # False: if not

imbalanced_data = True  # True: if is used imbalanced data 
                        # False: if not

selected_model = choose_models('GBC',params=None) 

selected_params = {'learning_rate': [0.01],
                    'loss': ['log_loss'],
                    'max_depth': [8], 
                    'max_features': [0.3], 
                    'min_samples_leaf': [250], 
                    'n_estimators': [100]}

data = pd.read_csv(os.getcwd()+'/data/processed/data_cleaned.csv').drop(columns=['Unnamed: 0'])

target = 'is_safe'

# ==================================================================
####################################################################

models_generator(data,target,selected_model,selected_params,
file_name='training_metrics.csv',dir_file='model/model_metrics',dir_model_file='model',scaling = True,
scoring = { "AUC": "roc_auc","Accuracy": make_scorer(accuracy_score)}, balancing = False)