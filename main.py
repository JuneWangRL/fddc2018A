# -*- coding: utf-8 -*-
from code_xg2 import market_data_pre
from code_xg1 import preprocess_helper, preprocess_helper1, feature_extraction, assign_before

import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.externals import joblib
import lightgbm as lgb
import datetime
from statsmodels.tsa.stattools import acf, pacf,adfuller
import statsmodels.api as sm
import math
import warnings
import csv

if __name__ == "__main__":
	type_set = [ 1,  2,  3,  4,  6,  7,  9, 10, 12, 13, 14, 16, 17, 18, 21, 19,  8, 20, 23,  0,  5, 24, 11, 25, 26, 27]
	market_data = pd.read_excel('/home/fddc1_data/Market Data.xlsx')  #!!!
	market_data.drop_duplicates(subset=['TICKER_SYMBOL', 'END_DATE_'], inplace=True)
	market_data1 = market_data[market_data['END_DATE_'] == '2018/5/31']
	market_data1['TYPE'] = market_data1.apply(market_data_pre, axis=1)
	market_data1 = market_data1[['TICKER_SYMBOL', 'MARKET_VALUE', 'TYPE']]

	# For the data preprocessing and generating the input features
	print ('Read income')
	income_business = pd.read_excel('/home/fddc1_data/financial_data/Income Statement.xls', 'General Business') #!!!
	income_bank = pd.read_excel('/home/fddc1_data/financial_data/Income Statement.xls', 'Bank')
	income_securities = pd.read_excel('/home/fddc1_data/financial_data/Income Statement.xls', 'Securities')
	income_insurance = pd.read_excel('/home/fddc1_data/financial_data/Income Statement.xls', 'Insurance')

	print ('Preprocess income')
	income_business_update = preprocess_helper(income_business)
	income_bank_update = preprocess_helper(income_bank)
	income_securities_update = preprocess_helper(income_securities)
	income_insurance_update = preprocess_helper(income_insurance)

	print ('Read balance')
	balance_business = pd.read_excel('/home/fddc1_data/financial_data/Balance Sheet.xls', 'General Business') #!!!
	balance_bank = pd.read_excel('/home/fddc1_data/financial_data/Balance Sheet.xls', 'Bank')
	balance_securities = pd.read_excel('/home/fddc1_data/financial_data/Balance Sheet.xls', 'Securities')
	balance_insurance = pd.read_excel('/home/fddc1_data/financial_data/Balance Sheet.xls', 'Insurance')

	print ('Preprocess balance')
	balance_business_update = preprocess_helper1(balance_business)
	balance_bank_update = preprocess_helper1(balance_bank)
	balance_securities_update = preprocess_helper1(balance_securities)
	balance_insurance_update = preprocess_helper1(balance_insurance)

	print ('Concatenate')
	income_sum = pd.concat([income_business_update, income_bank_update, income_securities_update, income_insurance_update])
	balance_sum = pd.concat([balance_business_update, balance_bank_update, balance_securities_update, balance_insurance_update])

	print ('Merge')
	income_with_market_value = income_sum.merge(market_data1, on='TICKER_SYMBOL')
	data_p_summary = income_with_market_value.merge(balance_sum, on=['TICKER_SYMBOL', 'END_DATE'])
	data_p_summary['END_DATE_STR'] = data_p_summary['END_DATE'].str.split(',')
	data_p_summary.fillna(0, inplace=True)

	print ('Adding feature')
	preprocessed_data_summary = feature_extraction(data_p_summary)
	#处理后的数据preprocessed_data_summary
	preprocessed_data_summary.to_excel('/home/118_118/temp/preprocessed_data_summary.xlsx')
	
	submit_path = '/home/fddc1_data/predict_list.csv'
	submit_data = pd.read_csv(submit_path, header=None)

	submit_data.columns = ['key']
	submit_data['TICKER_SYMBOL'] = submit_data.apply(lambda x: str(x['key'][0:6]), axis=1)

	data_summary_path = '/home/118_118/temp/preprocessed_data_summary.xlsx'
	data_sum = pd.read_excel(data_summary_path)
	data_sum['TICKER_SYMBOL'] = data_sum.apply(lambda x: str(x['TICKER_SYMBOL']).zfill(6), axis=1)
	
	# 1. xgboost
	print ('xgboost prediction')
	in1 = pd.read_csv('/home/fddc1_data/predict_list.csv', header=None)
	in1.columns = ['TICKER_SYMBOL_']
	in1['TICKER_SYMBOL'] = in1.apply(lambda x: x['TICKER_SYMBOL_'][0:6], axis=1)
	submit_symbol = in1['TICKER_SYMBOL'].unique()
	xgbsummary = pd.DataFrame()
	for t in type_set:
		sub_type_data = data_sum[data_sum['TYPE'] == t]
		sub_type_data.sort_values(by=['END_DATE'], inplace=True)
		length = sub_type_data.TICKER_SYMBOL.count()

		unique_symbols = sub_type_data['TICKER_SYMBOL'].unique()
		sub_type_data_in = sub_type_data[['isQ1', 'isS1', 'isQ3', 'isA', 'LAST_Q_REVENUE', 'LAST_YEAR_REVENUE', 'LAST_Q_ASSETS', 'LAST_YEAR_ASSETS']]
		sub_type_data_out = sub_type_data[['REVENUE']]
		model = XGBRegressor(learning_rate =0.1,n_estimators=2200,max_depth=3,min_child_weight=7,gamma=0,subsample=0.8,colsample_bytree=1,reg_alpha=0.005, objective='reg:linear')
		model.fit(sub_type_data_in, np.array(sub_type_data_out).ravel())
		model_name = 'clf' + str(t)
		inter = (x for x in unique_symbols if x in submit_symbol)
		for i in inter:
			tmp = sub_type_data[sub_type_data['TICKER_SYMBOL'] == i]
			predict_data = pd.DataFrame()
			result = pd.DataFrame()
			predict_data['isQ1'] = [0]
			predict_data['isS1'] = [1]
			predict_data['isQ3'] = [0]
			predict_data['isA'] = [0]
			predict_data['LAST_Q_REVENUE'] = tmp[tmp['END_DATE'] == '2018-03-31'].REVENUE.values
			predict_data['LAST_YEAR_REVENUE'] = tmp[tmp['END_DATE'] == '2017-06-30'].REVENUE.values
			predict_data['LAST_Q_ASSETS'] = tmp[tmp['END_DATE'] == '2018-03-31'].T_ASSETS.values
			predict_data['LAST_YEAR_ASSETS'] = tmp[tmp['END_DATE'] == '2017-06-30'].T_ASSETS.values
			predict_data.fillna(0, inplace=True)
			pre = model.predict(predict_data)
			result['TICKER_SYMBOL'] = [i]
			result['PREDICTIONS'] = round(pre[0]/1000000,2)
			xgbsummary = pd.concat([xgbsummary, result])
	xgbsummary['PREDICTIONS'] = xgbsummary.apply(lambda x: x['PREDICTIONS'], axis=1)
	xgbsummary.to_csv('/home/118_118/temp/xgbtmp.csv', index=None)
	re_xgb = in1.merge(xgbsummary, on='TICKER_SYMBOL')
	re_xgb = re_xgb[['TICKER_SYMBOL_', 'PREDICTIONS']]
	re_xgb.columns = ['TICKER_SYMBOL', 'XGB_PREDICTIONS']
	re_xgb.to_csv('/home/118_118/temp/xgbresult.csv', index=False)

# 2. lightgbm
	print('lgb prediction')
	params = {
	'objective':'regression',
	'boosting_type': 'gbdt',
	'metric': {'huber'},
	'learning_rate':0.05, 
	'n_estimators':2000,
	'bagging_fraction': 0.9,
	'bagging_freq': 6, 
	'feature_fraction': 0.9,
	'min_data_in_leaf':14, 
	'num_leaves':31,
	'max_depth': 16,
	'max_bin': 280
	}

	in1 = pd.read_csv('/home/fddc1_data/predict_list.csv', header=None)
	in1.columns = ['TICKER_SYMBOL_']
	in1['TICKER_SYMBOL'] = in1.apply(lambda x: x['TICKER_SYMBOL_'][0:6], axis=1)
	submit_symbol = in1['TICKER_SYMBOL'].unique()
	lgbsummary = pd.DataFrame()
	for t in type_set:
		sub_type_data = data_sum[data_sum['TYPE'] == t]
		sub_type_data.sort_values(by=['END_DATE'], inplace=True)
		sub_type_data_train = sub_type_data.loc[(sub_type_data['END_DATE'] <= '2018-03-31')]
		sub_type_data_test = sub_type_data.loc[sub_type_data['END_DATE'] > '2017-03-31']
		length = sub_type_data.TICKER_SYMBOL.count()
		unique_symbols = sub_type_data['TICKER_SYMBOL'].unique()

		sub_type_data_in_train = sub_type_data_train[['isQ1', 'isS1', 'isQ3', 'isA', 'LAST_Q_REVENUE', 'LAST_YEAR_REVENUE', 'LAST_Q_ASSETS', 'LAST_YEAR_ASSETS']]
		sub_type_data_out_train = sub_type_data_train[['REVENUE']]

		sub_type_data_in_test = sub_type_data_test[['isQ1', 'isS1', 'isQ3', 'isA', 'LAST_Q_REVENUE', 'LAST_YEAR_REVENUE', 'LAST_Q_ASSETS', 'LAST_YEAR_ASSETS']]
		sub_type_data_out_test = sub_type_data_test[['REVENUE']]
		
		lgb_train = lgb.Dataset(sub_type_data_in_train, label=np.array(sub_type_data_out_train).ravel())
		lgb_eval = lgb.Dataset(sub_type_data_in_test, label=np.array(sub_type_data_out_test).ravel(), reference=lgb_train)

		# train
		gbm = lgb.train(params,lgb_train,num_boost_round=500,valid_sets=lgb_eval,early_stopping_rounds=50)
		model_name = 'clf' + str(t)
		inter = (x for x in unique_symbols if x in submit_symbol)
		for i in inter:
			tmp = sub_type_data[sub_type_data['TICKER_SYMBOL'] == i]
			predict_data = pd.DataFrame()
			result = pd.DataFrame()
			predict_data['isQ1'] = [0]
			predict_data['isS1'] = [1]
			predict_data['isQ3'] = [0]
			predict_data['isA'] = [0]
			predict_data['LAST_Q_REVENUE'] = tmp[tmp['END_DATE'] == '2018-03-31'].REVENUE.values
			predict_data['LAST_YEAR_REVENUE'] = tmp[tmp['END_DATE'] == '2017-06-30'].REVENUE.values
			predict_data['LAST_Q_ASSETS'] = tmp[tmp['END_DATE'] == '2018-03-31'].T_ASSETS.values
			predict_data['LAST_YEAR_ASSETS'] = tmp[tmp['END_DATE'] == '2017-06-30'].T_ASSETS.values
			predict_data.fillna(0, inplace=True)
			pre = gbm.predict(predict_data, num_iteration=gbm.best_iteration)
			result['TICKER_SYMBOL'] = [i]
			result['PREDICTIONS'] = round(pre[0]/1000000,2)
			lgbsummary = pd.concat([lgbsummary, result])
	lgbsummary['PREDICTIONS'] = lgbsummary.apply(lambda x: x['PREDICTIONS'], axis=1)
	lgbsummary.to_csv('/home/118_118/temp/lgbtmp.csv', index=None)
	re_lgb = in1.merge(lgbsummary, on='TICKER_SYMBOL')
	re_lgb = re_lgb[['TICKER_SYMBOL_', 'PREDICTIONS']]
	re_lgb.columns=['TICKER_SYMBOL', 'LGB_PREDICTIONS']
	re_lgb.to_csv('/home/118_118/temp/lgbresult.csv', index=False)

# 3. gbdt
	in1 = pd.read_csv('/home/fddc1_data/predict_list.csv', header=None)
	in1.columns = ['TICKER_SYMBOL_']
	in1['TICKER_SYMBOL'] = in1.apply(lambda x: x['TICKER_SYMBOL_'][0:6], axis=1)
	submit_symbol = in1['TICKER_SYMBOL'].unique()

	gbdtsummary = pd.DataFrame()
	for t in type_set:
		sub_type_data = data_sum[data_sum['TYPE'] == t]
		sub_type_data.sort_values(by=['END_DATE'], inplace=True)
		length = sub_type_data.TICKER_SYMBOL.count()
		unique_symbols = sub_type_data['TICKER_SYMBOL'].unique()
		sub_type_data_in = sub_type_data[['isQ1', 'isS1', 'isQ3', 'isA', 'LAST_Q_REVENUE', 'LAST_YEAR_REVENUE', 'LAST_Q_ASSETS', 'LAST_YEAR_ASSETS']]
		sub_type_data_out = sub_type_data[['REVENUE']]
		model = GradientBoostingRegressor(learning_rate=0.1, max_depth=3,min_samples_split=15, n_estimators=1500, loss='ls')
		model.fit(sub_type_data_in, np.array(sub_type_data_out).ravel())
		model_name = 'clf' + str(t)
		inter = (x for x in unique_symbols if x in submit_symbol)
		for i in inter:
			tmp = sub_type_data[sub_type_data['TICKER_SYMBOL'] == i]
			predict_data = pd.DataFrame()
			result = pd.DataFrame()
			predict_data['isQ1'] = [0]
			predict_data['isS1'] = [1]
			predict_data['isQ3'] = [0]
			predict_data['isA'] = [0]
			predict_data['LAST_Q_REVENUE'] = tmp[tmp['END_DATE'] == '2018-03-31'].REVENUE.values
			predict_data['LAST_YEAR_REVENUE'] = tmp[tmp['END_DATE'] == '2017-06-30'].REVENUE.values
			predict_data['LAST_Q_ASSETS'] = tmp[tmp['END_DATE'] == '2018-03-31'].T_ASSETS.values
			predict_data['LAST_YEAR_ASSETS'] = tmp[tmp['END_DATE'] == '2017-06-30'].T_ASSETS.values
			predict_data.fillna(0, inplace=True)
			pre = model.predict(predict_data)
			result['TICKER_SYMBOL'] = [i]
			result['PREDICTIONS'] = round(pre[0]/1000000,2)
			gbdtsummary = pd.concat([gbdtsummary, result])
	gbdtsummary['PREDICTIONS'] = gbdtsummary.apply(lambda x: x['PREDICTIONS'], axis=1)
	gbdtsummary.to_csv('/home/118_118/temp/gbdttmp.csv', index=None)
	re_gbdt = in1.merge(gbdtsummary, on='TICKER_SYMBOL')
	re_gbdt = re_gbdt[['TICKER_SYMBOL_', 'PREDICTIONS']]
	re_gbdt.columns = ['TICKER_SYMBOL', 'GBDT_PREDICTIONS']
	re_gbdt.to_csv('/home/118_118/temp/gbdtresult.csv', index=False)

#4.arima
warnings.filterwarnings("ignore")
print('ARIMA MODEL PREDICTION')
submit=pd.read_csv('/home/fddc1_data/predict_list.csv',header=None,names=['Code'])
total_income_sheet=pd.DataFrame(columns=['TICKER_SYMBOL','EXCHANGE_CD','PUBLISH_DATE','END_DATE','FISCAL_PERIOD','REVENUE'])
def load_data(ind_name):

    ind_income_sheet= pd.read_excel(r'/home/fddc1_data/financial_data/Income Statement.xls',
                                    sheet_name=ind_name,
                                    dtype={'TICKER_SYMBOL':str})
    ind_income_sheet=ind_income_sheet.loc[:,['TICKER_SYMBOL','EXCHANGE_CD','PUBLISH_DATE','END_DATE','FISCAL_PERIOD','REVENUE']]
    return ind_income_sheet

ge_income_sheet=load_data('General Business')
bank_income_sheet=load_data('Bank')
insurance_income_sheet=load_data('Insurance')
securities_income_sheet=load_data('Securities')
print('processing data')
total_income_sheet=pd.concat([ge_income_sheet,bank_income_sheet,insurance_income_sheet,securities_income_sheet])
total_income_sheet['Code']=total_income_sheet['TICKER_SYMBOL']+"."+total_income_sheet['EXCHANGE_CD']
submit_income_sheet=total_income_sheet[total_income_sheet['Code'].isin(submit['Code'])]
submit_income_sheet['PUBLISH_DATE_use']=pd.to_datetime(submit_income_sheet['PUBLISH_DATE'])
submit_income_sheet['END_DATE_use']=pd.to_datetime(submit_income_sheet['END_DATE'])
submit_company_sorted=submit_income_sheet.sort_values(axis = 0,ascending = True,by = ['Code','FISCAL_PERIOD','END_DATE','PUBLISH_DATE'])
submit_company_reindex=submit_company_sorted.reset_index(drop=True)
submit_company_dup_id=submit_company_reindex.duplicated(subset=['Code','FISCAL_PERIOD','END_DATE'])
submit_company_dup=submit_company_reindex[submit_company_dup_id.values == False]
income_sheet=submit_company_dup
income_sheet['report_year']=income_sheet['END_DATE_use'].apply(lambda t:t.year)
del submit_company_dup_id,submit_company_reindex,submit_company_sorted
del submit_company_dup,submit_income_sheet,total_income_sheet

###-----将报告数据改成季度营收数据----------
companys_quarterly=pd.DataFrame(columns=['report_year', 'FISCAL_PERIOD', 'REVENUE', 'TICKER_SYMBOL'])
submit['TICKER_SYMBOL']=[x[:6] for x in submit['Code']]
company_name_list=list(submit['TICKER_SYMBOL'].values)
def get_company_quarterly(company_name):
    global companys_quarterly
    company_df=income_sheet[income_sheet['TICKER_SYMBOL']==company_name]
    company_pivot=company_df.pivot(index='report_year',columns='FISCAL_PERIOD',
                                   values='REVENUE')
    company_pivot['q1']=company_pivot[3]
    company_pivot['q2']=company_pivot[6]-company_pivot[3]
    company_pivot['q3']=company_pivot[9]-company_pivot[6]
    company_pivot['q4']=company_pivot[12]-company_pivot[9]
    company_pivot_m=company_pivot.drop(columns=[3,6,9,12])
    company_series=company_pivot_m.stack()
    company_quarter=pd.DataFrame(index=company_series.index)
    company_quarter['REVENUE']=company_series.values
    company_quarter['TICKER_SYMBOL']=company_name
    company_quarter.reset_index(inplace=True)
    companys_quarterly=pd.concat([companys_quarterly,company_quarter])
    #print(company_name)
    return company_name
    
company_quarterly=[get_company_quarterly(x) for x in company_name_list]
print('predict using ARIMA model')
######-------预测---------#####
def station_test(ts):    
    dftest=adfuller(ts,maxlag=10)
    df_p=dftest[1]
    if df_p>=0.05:
        stationarity=False
    elif df_p<0.05:
        stationarity=True
    return stationarity

def ARMA_MODEL(timeseries):
    order=sm.tsa.arma_order_select_ic(timeseries,max_ar=3,max_ma=3,ic='bic')['bic_min_order'] 
    temp_model= sm.tsa.ARMA(timeseries,order).fit()
    pred=temp_model.forecast(1)

    return pred[0][-1]

def decompose2(timeseries):
    station_diff=False
    diff1=timeseries.diff(1).dropna()
    station_diff1=station_test(diff1)
    if station_diff1:
        pred=ARMA_MODEL(diff1)+timeseries.values[-1]
        station_diff=True
    else:      
        diff4=diff1.diff(4).dropna()
        station_diff4=station_test(diff4)
        if station_diff4:
            pred=ARMA_MODEL(diff4) + timeseries.values[-1] + timeseries.values[-4] - timeseries.values[-5]
            station_diff=True
        else:
            pred=0
            station_diff=False
    return station_diff,pred
    
def ARIMA_pred(company_name):
    company_quarter=companys_quarterly[companys_quarterly['TICKER_SYMBOL']==company_name]    
    timeseries=company_quarter['REVENUE']
    last_quarter=timeseries.values[-1]
    stationarity=station_test(timeseries)
    if stationarity:
        pred=ARMA_MODEL(timeseries)
        stat=True
    else:
        stat,pred=decompose2(timeseries)
    return (company_name,stat,pred,last_quarter)

result=[]
for x in company_name_list:
    try:
        pred_result=ARIMA_pred(x)
        result.append(pred_result)
    except:
        continue
    
pred_arima=pd.DataFrame()
pred_arima['TICKER_SYMBOL']=[x[0] for x in result]
pred_arima['arima_stat']=[x[1] for x in result]
pred_arima['arima_pred_q1']=[x[2] for x in result]
print('predict using cycle model')
######-------cyc预测---------#####

def cyclicity_pred(company_name):
    company_quarter=companys_quarterly[companys_quarterly['TICKER_SYMBOL']==company_name]
    company_quarter_3year=company_quarter[company_quarter['report_year']>2014]
    last_quarter=company_quarter_3year['REVENUE'].values[-1]
    if len(company_quarter_3year)<12:
        cyclicity=False
        forcast=last_quarter
        return cyclicity,forcast
    else:
        ACF1 = acf(company_quarter['REVENUE'])[1]
        ACF4 = acf(company_quarter['REVENUE'])[4]
        if ACF1>ACF4:
            cyclicity=False
            forcast=list(company_quarter['REVENUE'])[-1]
        else:
            cyclicity=True
            forcast=list(company_quarter['REVENUE'])[-4]
        last_quarter=list(company_quarter_3year['REVENUE'])[-1]
        return company_name,cyclicity,forcast,last_quarter

pred_result=pd.DataFrame()
pred_result['cyc']=[cyclicity_pred(x) for x in company_name_list]
pred_result['TICKER_SYMBOL']=[x[0] for x in pred_result['cyc']]

pred_result=pd.merge(submit,pred_result,on='TICKER_SYMBOL',how='left')
pred_result['cyc_pred_second']=[x[2] for x in pred_result['cyc']]
pred_result['last_quarter']=[x[2] for x in pred_result['cyc']]
pred_result['cyc_pred']=pred_result['cyc_pred_second']+pred_result['last_quarter']

pred_result=pd.merge(pred_result,pred_arima,on='TICKER_SYMBOL',how='left')

print('fillna arima result')
def fillarima(i):
    arima_true=pred_result['arima_stat'][i]
    arima_pred=pred_result['arima_pred_q1'][i]
    cyc_pred=pred_result['cyc_pred_second'][i]          
    if arima_true:
        arima_pred_m=arima_pred
    else:
        arima_pred_m=cyc_pred
    if np.isnan(arima_pred):
        arima_pred_m=cyc_pred
    if arima_pred<0:
        arima_pred_m=cyc_pred*1.3/1000000      
    return arima_pred_m

pred_result['arima_pred_m']=[fillarima(i) for i in range(len(pred_result))]
pred_result['arima_pred']=pred_result['arima_pred_m']+pred_result['last_quarter']

pred_result['arima_pred']=pred_result['arima_pred']

nowtime=datetime.datetime.now()
year=nowtime.year
month=nowtime.month
day=nowtime.day
hour=nowtime.hour
minuate=nowtime.minute
second=nowtime.second

print('write result to arimaresult')
arima_result=pd.DataFrame()
arima_result=pred_result[['Code','arima_pred']]
#arima_result.to_csv(r'submit/submit_%s%s%s_%s%s%s.csv' %(year,month,day,hour,minuate,second),
#                 header=False,index=False)
arima_result.columns=['TICKER_SYMBOL', 'ARIMA_PREDICTIONS']
arima_result['ARIMA_PREDICTIONS']=round(arima_result['ARIMA_PREDICTIONS']/1000000,2)
arima_result.to_csv(r'/home/118_118/temp/arimaresult.csv',index=False)

#数据处理
xgbresult = pd.read_csv('/home/118_118/temp/xgbresult.csv')
csv_reader=csv.reader(open("/home/118_118/temp/xgbresult.csv"))
stockCode=list()
for row in csv_reader:
	stockCode.append(row[0][0:6])
xgbresult['TICKER_SYMBOL']=stockCode[1:]

lgbresult = pd.read_csv('/home/118_118/temp/lgbresult.csv')
csv_reader=csv.reader(open("/home/118_118/temp/lgbresult.csv"))
stockCode=list()
for row in csv_reader:
	stockCode.append(row[0][0:6])
lgbresult['TICKER_SYMBOL']=stockCode[1:]

gbdtresult = pd.read_csv('/home/118_118/temp/gbdtresult.csv')
csv_reader=csv.reader(open("/home/118_118/temp/gbdtresult.csv"))
stockCode=list()
for row in csv_reader:
	stockCode.append(row[0][0:6])
gbdtresult['TICKER_SYMBOL']=stockCode[1:]

arimaresult = pd.read_csv('/home/118_118/temp/arimaresult.csv')
csv_reader=csv.reader(open("/home/118_118/temp/arimaresult.csv"))
stockCode=list()
for row in csv_reader:
	stockCode.append(row[0][0:6])
arimaresult['TICKER_SYMBOL']=stockCode[1:]

xgbresult['TICKER_SYMBOL']=xgbresult['TICKER_SYMBOL'].apply(lambda t:int(t))					
lgbresult['TICKER_SYMBOL']=lgbresult['TICKER_SYMBOL'].apply(lambda t:int(t))					
gbdtresult['TICKER_SYMBOL']=gbdtresult['TICKER_SYMBOL'].apply(lambda t:int(t))					
arimaresult['TICKER_SYMBOL']=arimaresult['TICKER_SYMBOL'].apply(lambda t:int(t))					

xgbresult.rename(columns={'XGB_PREDICTIONS':'PREDICTIONS'},inplace = True)
lgbresult.rename(columns={'LGB_PREDICTIONS':'PREDICTIONS'},inplace = True)
gbdtresult.rename(columns={'GBDT_PREDICTIONS':'PREDICTIONS'},inplace = True)
arimaresult.rename(columns={'ARIMA_PREDICTIONS':'PREDICTIONS'},inplace = True)

#数据读入完毕
#.....每个行业取在测试集上最好的模型
market_data1=market_data1[market_data1['TICKER_SYMBOL'].isin(stockCode[1:])]
names=locals()
for i in range(28):
	names["type%i"%i]=list(market_data1[market_data1['TYPE']==i]['TICKER_SYMBOL'])

result_type=pd.DataFrame(columns=['TICKER_SYMBOL', 'PREDICTIONS'])
type_model=["arima","arima","lgb","lgb","arima","lgb","gbdt","xgb","gbdt","gbdt","arima","xgb","gbdt","lgb","xgb","","lgb","gbdt","xgb","gbdt","xgb","arima"," ","arima","arima","arima","arima","arima"]
for i in range(28):
	if type_model[i]=="xgb":
		df=xgbresult
	if type_model[i]=="lgb":
		df=lgbresult
	if type_model[i]=="gbdt":
		df=gbdtresult
	if type_model[i]=="arima":
		df=arimaresult
	names["df%i"%i]=df[df['TICKER_SYMBOL'].isin(names["type%i"%i])]

for i in range(28):
	result_type=pd.concat([result_type,names["df%i"%i]],axis=0) 		
result_type=result_type.reset_index(drop=True)
result = pd.read_csv('/home/118_118/temp/xgbresult.csv')
result['code']=result['TICKER_SYMBOL'].apply(lambda t:int(t[0:6]))	
result['XGB_PREDICTIONS']=0
for i in range(len(result)):
	for j in range(len(result_type)):
		if result.iloc[i,2]==result_type.iloc[j,0]:
			result.iloc[i,1]=result_type.iloc[j,1]
del result['code']
result.to_csv(("/home/118_118/submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), header=None, index=False)




