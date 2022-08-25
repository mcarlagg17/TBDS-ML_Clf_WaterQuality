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

unsupervised = True     # True: if is used unsupervised estimators 
                        # False: if not

supervised = True       # True: if is used supervised estimators 
                        # False: if not

deep_learning = False   # True: if is used deep learning 
                        # False: if not

imbalanced_data = True  # True: if is used imbalanced data 
                        # False: if not

new_model = choose_models('RFC') 

# ==================================================================
####################################################################