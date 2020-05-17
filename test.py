#%%
import pickle
import pandas as pd
#%%
with open('/Users/ziaoyan/PycharmProjects/比赛记录切片/result.pickle','rb') as load_data:
    result = pickle.load(load_data)
with open('/Users/ziaoyan/PycharmProjects/比赛记录切片/record_slice.pickle','rb') as load_data:
    data = pickle.load(load_data)
#%%
type(data)
data_df=pd.DataFrame(data)
#%%
data_df_building=data_df.iloc[:,262:292]
data_df_GOLDEXP=data_df.iloc[:,292:294]
data_df_NOHero =data_df.iloc[:,262:,]
#%%
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pl
#%%
Xt, Xv, yt, yv = train_test_split(data_df,result, test_size=0.2, random_state=10)
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(Xt, yt)

Xt_building, Xv_building,yt_building,yv_building = train_test_split(data_df_building,result,test_size=0.3,random_state=10)
xgb_model_building = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model_building.fit(Xt_building, yt_building)

Xt_NOHero, Xv_NOHero,yt_NOHero,yv_NOHero = train_test_split(data_df_NOHero,result,test_size=0.3,random_state=10)
xgb_model_NoHero = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model_NoHero.fit(Xt_NOHero, yt_NOHero)

Xt_GolDExp, Xv_GolDExp,yt_GolDExp,yv_GolDExp = train_test_split(data_df_GOLDEXP,result,test_size=0.3,random_state=10)
xgb_model_GoldExp = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model_GoldExp.fit(Xt_GolDExp, yt_GolDExp)
#%%
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import metrics
predictions = xgb_model.predict(Xv)
actuals = yv
#fpr,tpr,thresholds = metrics.roc_curve(actuals,predictions,pos_label=2)
#metrics.auc(fpr,tpr)
print(accuracy_score(actuals, predictions))
#print(metrics.auc(fpr,tpr))
print(metrics.roc_auc_score(predictions,yv))

#%%
from sklearn.metrics import accuracy_score
from sklearn import metrics
predictions = xgb_model_GoldExp.predict(Xv_GolDExp)
actuals = yv_GolDExp
#fpr,tpr,thresholds = metrics.roc_curve(actuals,predictions,pos_label=2)
#metrics.auc(fpr,tpr)
print(accuracy_score(actuals, predictions))
#print(metrics.auc(fpr,tpr))
print(metrics.roc_auc_score(predictions,yv))
#%%
from sklearn.metrics import accuracy_score
from sklearn import metrics
predictions = xgb_model_NoHero.predict(Xv_NOHero)
actuals = yv_NOHero
#fpr,tpr,thresholds = metrics.roc_curve(actuals,predictions,pos_label=2)
#metrics.auc(fpr,tpr)
print(accuracy_score(actuals, predictions))
#print(metrics.auc(fpr,tpr))
print(metrics.roc_auc_score(predictions,yv))
#%%
from sklearn.metrics import accuracy_score
from sklearn import metrics
predictions = xgb_model_GoldExp.predict(Xv_building)
actuals = yv_building
#fpr,tpr,thresholds = metrics.roc_curve(actuals,predictions,pos_label=2)
#metrics.auc(fpr,tpr)
print(accuracy_score(actuals, predictions))
#print(metrics.auc(fpr,tpr))
print(metrics.roc_auc_score(predictions,yv))