#%%
import pickle
import pandas as pd
#%%
with open('/Users/ziaoyan/Desktop/result.pickle','rb') as load_data:
    result = pickle.load(load_data)
with open('/Users/ziaoyan/Desktop/record_slice.pickle','rb') as load_data:
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
#save model
pickle.dump(xgb_model, open("xgboost.pickle.dat", "wb"))
pickle.dump(xgb_model_GoldExp, open("xgboost_GoldExp.pickle.dat", "wb"))
pickle.dump(xgb_model_building, open("xgboost_Building.pickle.dat", "wb"))
pickle.dump(xgb_model_NoHero, open("xgboost_NoHero.pickle.dat", "wb"))


#%%
xgb_model_new=pickle.load(open("xgboost.pickle.dat", "rb"))
predictions = xgb_model_new.predict(Xv)
actuals = yv
#fpr,tpr,thresholds = metrics.roc_curve(actuals,predictions,pos_label=2)
#metrics.auc(fpr,tpr)
print(accuracy_score(actuals, predictions))
#print(metrics.auc(fpr,tpr))
print(metrics.roc_auc_score(predictions,actuals))
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
print(metrics.roc_auc_score(predictions,actuals))
#%%
from sklearn.metrics import accuracy_score
from sklearn import metrics
predictions = xgb_model_building.predict(Xv_building)
actuals = yv_building
#fpr,tpr,thresholds = metrics.roc_curve(actuals,predictions,pos_label=2)
#metrics.auc(fpr,tpr)
print(accuracy_score(actuals, predictions))
#print(metrics.auc(fpr,tpr))
print(metrics.roc_auc_score(predictions,actuals))
#%%
test_data=pd.read_csv('/Users/ziaoyan/Desktop/match_slice_new1.csv')
test_data_init=test_data.groupby(by='match_id').head(1)
#%%
test_data=test_data.drop(columns=['match_id','Unnamed: 0'])
test_data_init=test_data_init.drop(columns=['match_id','Unnamed: 0'])

#%%
result=test_data[['result']]
result_init=test_data_init[['result']]
#%%
test_data=test_data.drop(columns=['result'])
test_data_init=test_data_init.drop(columns=['result'])

#%%
columns_name=Xv.columns.tolist()
#%%
test_data.columns=columns_name
test_data_init.columns=columns_name

#%%
predictions = xgb_model.predict(test_data)
actuals = result
print(accuracy_score(actuals, predictions))
#%%
predictions = xgb_model.predict(test_data_init)
actuals = result_init
print(accuracy_score(actuals, predictions))

#%%
predictions_prob=xgb_model.predict_proba(test_data)
predictions_prob_win=predictions_prob[:,-1]
from matplotlib import pyplot
pyplot.hist(predictions_prob_win,bins=30)
pyplot.show()

#%%

#%%
test_df_building=test_data.iloc[:,262:292]
test_df_GOLDEXP=test_data.iloc[:,292:294]
test_df_NOHero =test_data.iloc[:,262:,]
#%%
record_train, record_test, result_train, result_test = train_test_split(test_data,result, test_size=0.2, random_state=10)
xgb_model_test = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model_test.fit(record_train, result_train)
#%%
record_train_Gold, record_test_Gold, result_train_Gold, result_test_Gold = train_test_split(test_df_GOLDEXP,result, test_size=0.2, random_state=10)
xgb_model_test_Gold = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model_test_Gold.fit(record_train_Gold, result_train_Gold)
#%%
predictions = xgb_model_GoldExp.predict(test_df_GOLDEXP)
actuals = result
#fpr,tpr,thresholds = metrics.roc_curve(actuals,predictions,pos_label=2)
#metrics.auc(fpr,tpr)
print(accuracy_score(actuals, predictions))
#print(metrics.auc(fpr,tpr))
print(metrics.roc_auc_score(predictions,actuals))
#%%

predictions = xgb_model_NoHero.predict(Xv_NOHero)
actuals = yv_NOHero
#fpr,tpr,thresholds = metrics.roc_curve(actuals,predictions,pos_label=2)
#metrics.auc(fpr,tpr)
print(accuracy_score(actuals, predictions))
#print(metrics.auc(fpr,tpr))
print(metrics.roc_auc_score(predictions,actuals))
#%%

predictions = xgb_model_building.predict(Xv_building)
actuals = yv_building
#fpr,tpr,thresholds = metrics.roc_curve(actuals,predictions,pos_label=2)
#metrics.auc(fpr,tpr)
print(accuracy_score(actuals, predictions))
#print(metrics.auc(fpr,tpr))
print(metrics.roc_auc_score(predictions,actuals))