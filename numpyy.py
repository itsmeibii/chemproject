from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, make_scorer
from sklearn.ensemble import IsolationForest
from collections import Counter
import pickle
from sklearn.feature_selection import SelectKBest, f_regression


# X,y = load_diabetes(return_X_y=True)
# #print(load_diabetes().DESCR)

# #Random Forest
# fpl = Pipeline([
#     ("scale", StandardScaler()),
#     ("model", RandomForestRegressor())
# ])
# # fmodel = fpl.fit(X,y)
# fpred = fmodel.predict(X)
# plt.scatter(fpred,y, color = 'green')


#KNeighbours
# knpl = Pipeline([
#     ("scale", StandardScaler()),
#     ("model", KNeighborsRegressor())
# ])
# model = knpl.fit(X,y)
# prediction = model.predict(X)
# plt.scatter(prediction,y, color = 'red')

#fpred is a lot better, more linear which means it is more consistent with the original targets
#The models above were used to predict the same data that they were trained upon
#--------------------------------------------------------------------------------------------------
#Now, I will attempt to train both models on half the data set, then predict the other half, and then plot those predictions against the targets, much like what I think cross validation is

#Random Forest
# truefmodel = fpl.fit(X[:221],y[:221])
# truefpred = truefmodel.predict(X[-221:])
# plt.scatter(truefpred,y[-221:],color = 'green')

# #KNeighbours
# truekmodel = knpl.fit(X[:221],y[:221])
# truekpred = truekmodel.predict(X[-221:])
# plt.scatter(truekpred,y[-221:], color = 'red')

#Have just now become aware of a train_test_split function that wouldve done this for me


#Now we see that they really arent all that much different without hyperparameter tuning. Random Forest was just a lot better at predicting when it is predicting the data that it was trained on
#This was a sort of half baked two fold cross validation that was done manually
#----------------------------------------------------------------------------------------------------------------
#Now I will perform cross validation how it is intended

#fpl.get_params()
# gridmodel = GridSearchCV(estimator = fpl,
#                          param_grid = {
#                              'model__n_estimators':[25,50,75,100,125,150,200], #Num of trees, decision makers
#                              'model__max_depth':[1,2,3,4,5,6,7,8,9,10] #Num of decisions made by each tree
                              
#                          },
#                          scoring = {'precision':make_scorer(precision_score), 'recall':make_scorer(recall_score)},
#                          refit = 'precision',
#                          cv = 3,
#                          n_jobs = -1)

# gridmodel.fit(X,y)
# data = pd.DataFrame(gridmodel.cv_results_) 
#print(data.loc[data['mean_test_score'].idxmax()])

#For Outliers
# outl = IsolationForest().fit(X)
# print(Counter(outl.predict(X)))
#70 Outliers, a large chunk of data


#The above takes forever to run
#----------------------------------------------------------------------------------------------------------------
#I do want to become familiar with train test split, so will play around with that
#Random Split
#X_train, X_temp, Y_train, Y_temp = train_test_split(X,y,test_size = 0.2,random_state = 42, test_size = 0.2 #20% of the data will be used for testing
                                                    #)
#Further split the testing set into validation and actual testing, with validation being for fine tuning
#How do i do that
#COME BACK TO THIS



#MAIN ACTUAL MODEL
#---------------------------------------------------------------------------------------------------------------------

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw


# Function to compute Bemis-Murcko scaffold
def bemis_murcko_scaffold(smiles, retasmol):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if retasmol:
        return scaffold
    else :
        return Chem.MolToSmiles(scaffold)


with open('./database.pickle', 'rb') as f:
    data = pickle.load(f)

df = pd.DataFrame(data).transpose()
df['Scaffold'] = df['smiles'].apply(bemis_murcko_scaffold, args=(True,))
# print(df.head())
# image = Draw.MolToImage(df.at[df.index[3],'Scaffold'])
# image.show()




#No duplicate scaffolds
unique = df['Scaffold'].unique()

#---------------------
# X = df.drop(columns=['calc_h'])
# Y = df['calc_h']
#---------------------


#Seperate training and testing scaffolds
train_scaff, test_scaff = train_test_split(unique,test_size= 0.2, random_state=42)
#Convert scaffolds back into molecules
train_data = df[df['Scaffold'].isin(train_scaff)]
test_data = df[df['Scaffold'].isin(test_scaff)]
#Pipeline wont work with objects and the scaffold was only needed for splitting
#<FEATURIZATION> ----------
smilescol = df['smiles']
df = df.drop(columns=['Scaffold', 'PubChemID', 'iupac', 'smiles', 'notes', 'nickname'])
pd.set_option('display.max_columns', None)
print(df.head())

#Prints columns as nums but types are objects, so fixing that
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = pd.to_numeric(df[column], errors='coerce')
#<FEATURIZATION /> -------

X_train = train_data.drop(columns=['calc_h'])
y_train = train_data['calc_h']

X_test = test_data.drop(columns=['calc_h'])
y_test = test_data['calc_h']
print(df.columns)

# pipeline = Pipeline([
#     ("scale", StandardScaler()),
#     ("feature_selection", SelectKBest(score_func=f_regression, k = 100)),
#     ("model", RandomForestRegressor())
# ])
# model = pipeline.fit(X_train,y_train)
# prediction = model.predict(X_train)
# plt.scatter(prediction,y_train, color = 'red')