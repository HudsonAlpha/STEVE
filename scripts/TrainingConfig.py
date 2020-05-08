
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from skopt.space import Real, Categorical, Integer
from xgboost import XGBClassifier

from ExtractFeatures import DEFAULT_MISSING

#define the training modes here
EXACT_MODE = 0 #really shouldn't be used, is here for historic reasons; cheap, but not the best practice
GRID_MODE = 1  #only uses defined grid parameters; this is semi-expensive, but good practice
BAYES_MODE = 2 #will search a range of parameters using BayesSearchCV to find params; this is expensive, should really only be done in high-dev mode

#the training methology to use
TRAINING_MODE = GRID_MODE

#this is the fraction of variants that are used for the final test; the remaining are used for training
TEST_FRACTION = 0.5

#if True, only use SUBSET_SIZE variants from each file (mostly for debugging)
USE_SUBSET = False
SUBSET_SIZE = 1000

#if True, manually remove features in MANUAL_FS_LABELS (these are historically unimportant features)
MANUAL_FS = True
MANUAL_FS_LABELS = ['CALL-ADO', 'CALL-AFO']

#if True, mark false positive as true positives and vice versa
FLIP_TP = True

#set up the classifiers, and enumerate which are enabled here; 
# changing to false removes the classifier from training which also reduces runtime
CLASSIFIERS = []

#Generally slight underperformance, but is a decent backup
ENABLE_RANDOMFOREST = True

#performs alright, but always seems to be below GradientBoostingClassifier and slower
ENABLE_ADABOOST = False

#historically, this one has been victorious
ENABLE_GRADIENTBOOST = True

#weirdly, this has always under-performed (both training speed is longer and results worse) compared to 
# GradientBoostingClassifier; we also have the issue that it isn't fully integrated into the sklearn 
# ecosystem; perhaps revisit in a couple months to see if it's fleshed out a little better
ENABLE_XGBOOST = False

#slightly beats GradientBoosting in hom. SNVs, but very slow to train; not worth CPU time IMO
ENABLE_EASYENSEMBLE = False

#this is an experimental mode in sklearn, it may change rapidly from version to version
ENABLE_HISTGRADIENTBOOST = True
if ENABLE_HISTGRADIENTBOOST:
    #make sure we can actually do what we're trying to do
    try:
        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingClassifier
    except:
        ENABLE_HISTGRADIENTBOOST = False

#here is where we put what each enable option indicates
#now enumerate the models as a tuple (
#   label - just a str label for outputs
#   default classifier - the base classifier we would use if CV is disabled
#   cross validation params - a dictionary of parameters to test during cross-validation
# )
# NOTE: some params have been pruned because they were very, very rarely selected during CV 
# and we want to reduce run-time when possible
if ENABLE_RANDOMFOREST:
    '''
    Random forest is generally relatively fast to train and a decent baseline.  It's almost always beaten by one
    of the other models.  However, it occasionally rises to task with a large enough model (i.e. n=500).  In general,
    if you do the grid search, most of your models will be 500 estimators with bootstrap and min_samples_split about
    half-and-half depending on the data type.
    '''
    CLASSIFIERS.append(
        ('RandomForest', RandomForestClassifier(random_state=0, class_weight='balanced', max_depth=4, n_estimators=200, min_samples_split=2, max_features='sqrt'),
        {
            'random_state' : [0],
            'class_weight' : ['balanced'],
            'n_estimators' : [500], #prior tests: 100, 200, 300
            'max_depth' : [6], #prior tests: 3, 4, 5
            'min_samples_split' : [2, 50],
            'max_features' : ['sqrt'],
            'bootstrap' : [True, False]
        },
        {
            'random_state' : Categorical([0]),
            'class_weight' : Categorical(['balanced']),
            'n_estimators' : Integer(200, 500), #basically always the max value
            'max_depth' : Integer(3, 6), #basically always 6, don't want to go higher due to over-fit
            'min_samples_split' : Integer(2, 50), #mostly at 50, but still mixed; TODO: raise the cap and/or convert to fractions?
            'max_features' : Categorical(['sqrt']),
            'bootstrap' : Categorical([True, False]),  #both used
            #'max_samples' : Real(0.1, 1.0, prior='uniform') #TODO: need to update scikit-learn to use this
        })
    )

if ENABLE_ADABOOST:
    CLASSIFIERS.append(
        '''
        Generally slower and worse results than other models, disabled after v1.
        '''
        #"The most important parameters are base_estimator, n_estimators, and learning_rate" - https://chrisalbon.com/machine_learning/trees_and_forests/adaboost_classifier/
        ('AdaBoost', AdaBoostClassifier(random_state=0, algorithm='SAMME.R', learning_rate=1.0, base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=200),
        {
            'random_state' : [0],
            'base_estimator' : [DecisionTreeClassifier(max_depth=2)], #prior tests: SVC(probability=True)
            'n_estimators' : [200, 300], #prior tests: 100
            'learning_rate' : [0.1, 1.0], #prior tests: 0.01
            'algorithm' : ['SAMME.R'] #prior tests: "SAMME"; didn't seem to matter much and SAMME.R is faster
        },
        {
            #Best params: OrderedDict([('algorithm', 'SAMME.R'), ('base_estimator', DecisionTreeClassifier(max_depth=3), ('learning_rate', 0.06563659204257402), ('n_estimators', 426), ('random_state', 0)])
            #ROC AUC: 0.999353
            'random_state' : Categorical([0]),
            'base_estimator' : Categorical([DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3)]), #prior tests: SVC(probability=True)
            'n_estimators' : Integer(300, 500), #prior tests: 100
            'learning_rate' : Real(0.0001, 1.0, prior='log-uniform'), #prior tests: 0.01
            'algorithm' : Categorical(['SAMME.R']) #prior tests: "SAMME"; didn't seem to matter much and SAMME.R is faster
        })
    )

if ENABLE_GRADIENTBOOST:
    '''
    This one almost always gets the best model.  In v2, changed to use early stopping which has the result of setting
    n_estimators to a very high value (1000) and fixing both the validation_fraction and n_iter_no_change to results
    derived from Bayes testing.  Max_depth and subsample were also always fixed in Bayes mode testing.  Learning rate 
    and splitting still show some variability depending on the data type, so we left a couple option in GridSearch.
    
    For some reason, it does seem to struggle with specifically homozygous SNVs.  I wonder if the frequency of FP is just
    low enough to make it a challenge for this model type.
    '''
    #" Most data scientist see number of trees, tree depth and the learning rate as most crucial parameters" - https://www.datacareer.de/blog/parameter-tuning-in-gradient-boosting-gbm/
    CLASSIFIERS.append(
        ('GradientBoosting', GradientBoostingClassifier(random_state=0, learning_rate=0.1, loss='exponential', max_depth=4, max_features='sqrt', n_estimators=200),
        {
            'random_state' : [0],
            'n_estimators' : [1000], #prior tests: 100, 200; OBSOLETE: since adding n_iter_no_change, just set to a big number
            'max_depth' : [6], #prior tests: 3, 4
            'learning_rate' : [0.05, 0.1, 0.5], #prior tests: 0.01, 0.2; from bayes mode, all results were in the 0.04-0.2 range with the occasional "high" rate near 0.5
            'loss' : ['exponential'], #prior tests: 'deviance'
            'max_features' : ['sqrt'],
            'min_samples_split' : [2, 15, 50], #mostly extremes in Bayes most, but adding 15 for middle-ground that was sometimes chosen
            'subsample' : [0.5], #when specifically checking subsample, the most consistent models (het snv/indels) both used 0.5; 0.9 was used on homs, but setting to 0.5 for reduced overfitting
            'validation_fraction' : [0.1], #every meaningful model went straight to lowest value
            'n_iter_no_change' : [20] #just need a single value, 20 seems decent overall
        },
        {
            #TODO: test the min_sample_split change below
            'random_state' : Categorical([0]),
            'n_estimators' : Categorical([1000]), #Used to be Integer(200, 500), but with n_iter_no_change, we can just set to a big number
            'max_depth' : Categorical([6]), # Used to be Integer(1, 6), but everything chose 6
            'learning_rate' : Real(0.0001, 0.5, prior='log-uniform'), #still shows variability, usually in range 0.01-0.5
            'loss' : Categorical(['exponential']), #prior tests: 'deviance' #just never seemed to out-perform exponential *shrug*
            'max_features' : Categorical(['sqrt']),
            'min_samples_split' : Integer(2, 50), #found we generally get both extremes 2 and 50, with some middle-ish ~15
            'subsample' : Categorical([0.5, 0.9, 1.0]),#Real(0.5, 0.9, prior='uniform'), #subsampling almost almost went straight to the max value but also commonly leads to overfitting
            'validation_fraction' : Categorical([0.1]), #Used to be "Real(0.1, 0.5, prior='uniform')", but 0.1 always chosen, so put bayes into other places
            'n_iter_no_change' : Categorical([20]) #used to be Integer(10, 20), but i think we can just safely set it to a single medium-sized value so fitting focuses on useful things
        })
    )

if ENABLE_XGBOOST:
    '''
    Despite the many positive reviews of XGB, it never really performed for me.  Also, early stopping is not easy to 
    implement in sklearn framework.  Leaving this here to revisit in the future, but is basically disabled by default 
    for now.
    '''
    CLASSIFIERS.append(
        ('XGBClassifier', XGBClassifier(random_state=0),
        {
            'random_state' : [0],
            'n_estimators' : [200], #default=100; prior tests: 100
            'max_depth' : [5, 6], #default=6; prior tests: 3, 4
            'learning_rate' : [0.1, 0.2], #default=0.3; prior tests: 0.3
            'subsample' : [0.5, 1.0], #default=1
            'tree_method' : ['approx'], #default=auto; prior tests: "hist"
            'objective' : ['binary:logistic', 'binary:logitraw'], #default=binary:logistic; prior tests 'binary:hinge'
            'missing' : [DEFAULT_MISSING] #NOTE: seems to be the only method that explicitly has a missing field; seems best practice to set it
        },
        {
            'random_state' : Categorical([0]),
            'n_estimators' : Integer(200, 500), #default=100; prior tests: 100
            'max_depth' : Integer(1, 8), #default=6; prior tests: 3, 4
            'learning_rate' : Real(0.0001, 0.5, prior='log-uniform'), #default=0.3; prior tests: 0.3
            'subsample' : Real(0.01, 1.0, prior='uniform'), #default=1
            'tree_method' : Categorical(['approx']), #default=auto; prior tests: "hist"
            'objective' : Categorical(['binary:logistic', 'binary:logitraw']), #default=binary:logistic; prior tests 'binary:hinge'
            'missing' : Categorical([DEFAULT_MISSING]) #NOTE: seems to be the only method that explicitly has a missing field; seems best practice to set it
        })
    )

if ENABLE_EASYENSEMBLE:
    '''
    Not a bad classifier, but really slow.  In v1, this was best for Hom. SNVs.  I think the Gradient or RF models are
    close enough that it isn't worth the CPU cycles to build these anymore.
    '''
    CLASSIFIERS.append(
        ('EasyEnsemble', EasyEnsembleClassifier(random_state=0, n_estimators=50),
        {
            'random_state' : [0],
            'n_estimators' : [40, 50, 75, 100] #prior tests: 10, 20, 30
        },
        {
            'random_state' : Categorical([0]),
            'n_estimators' : Integer(10, 100) #prior tests: 10, 20, 30
        })
    )

if ENABLE_HISTGRADIENTBOOST:
    '''
    In general, this seems to slightly under-perform compared to classic GradientBoosting above. However, the 
    results are relative and the training time on this is truly impressive. Prelim tests with 33 cores only 
    required ~1.5 hrs for ALL of the models for SS mode.  It's basically worth it to keep this no matter what 
    simple because training is ridiculously fast.  It might also be a good new baseline model to use just for
    speed purposes.  There are issues getting it to pass models with our criteria though, seems to have 
    consistency issues (could be over/under-fitting, hard to tell at this juncture).
    '''
    CLASSIFIERS.append(
        ('HistGradientBoosting', HistGradientBoostingClassifier(random_state=0),
        {
            'random_state' : [0],
            'learning_rate' : [0.05, 0.1], #high learning rates don't seem to work out very well
            'max_iter' : [1000],
            'max_leaf_nodes' : [31], #don't want to make this too high or it overfits
            'min_samples_leaf' : [200, 2000], #want this to be semi-high to avoid overfitting to a few variants
            'validation_fraction' : [0.1],
            'n_iter_no_change' : [20]
        },
        {
            'random_state' : Categorical([0]),
            'learning_rate' : Real(0.0001, 0.2, prior='log-uniform'),
            'max_iter' : Categorical([1000]),
            'max_leaf_nodes' : Integer(15, 255, prior='log-uniform'), #Categorical([7, 15, 31, 63, 127, None]),
            'min_samples_leaf' : Integer(20, 20000, prior='log-uniform'),
            'validation_fraction' : Categorical([0.1]),
            'n_iter_no_change' : Categorical([20])
        })
    )
