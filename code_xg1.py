import pandas as pd

income_path = '/home/fddc1_data/financial_data/Income Statement.xls'
market_value_path = '/home/118_118/temp/market_value_update.xlsx'
balance_path =  '/home/fddc1_data/financial_data/Balance Sheet.xls'

def preprocess_helper(input_data):
    input_data.sort_values(by=['TICKER_SYMBOL', 'END_DATE', 'PUBLISH_DATE'], inplace=True)
    input_data.drop_duplicates(subset=['TICKER_SYMBOL','END_DATE'], keep='last', inplace=True)
    input_data.sort_values(by=['TICKER_SYMBOL', 'END_DATE'], inplace=True)
    input_data_update = input_data[['TICKER_SYMBOL', 'END_DATE','REPORT_TYPE', 'REVENUE']]
    return input_data_update

def preprocess_helper1(input_data):
    input_data.sort_values(by=['TICKER_SYMBOL', 'END_DATE', 'PUBLISH_DATE'], inplace=True)
    input_data.drop_duplicates(subset=['TICKER_SYMBOL','END_DATE'], keep='last', inplace=True)
    input_data.sort_values(by=['TICKER_SYMBOL', 'END_DATE'], inplace=True)
    input_data_update = input_data[['TICKER_SYMBOL', 'END_DATE', 'T_ASSETS']]
    return input_data_update

def feature_extraction(input_data):
	input_data['isQ1'] = input_data.REPORT_TYPE.apply(lambda x: 1 if x == 'Q1' else 0)
	input_data['isS1'] = input_data.REPORT_TYPE.apply(lambda x: 1 if x == 'S1' else 0)
	input_data['isQ3'] = input_data.REPORT_TYPE.apply(lambda x: 1 if x == 'Q3' else 0)
	input_data['isA'] = input_data.REPORT_TYPE.apply(lambda x: 1 if x == 'A' else 0)
	input_data['LAST_Q_REVENUE'] = input_data.apply(lambda x: assign_before(x, 'Q1', input_data, 'REVENUE'), axis=1)
	input_data['LAST_YEAR_REVENUE'] = input_data.apply(lambda x: assign_before(x, 'A', input_data, 'REVENUE'), axis=1)
	input_data['LAST_Q_ASSETS'] = input_data.apply(lambda x: assign_before(x, 'Q1', input_data, 'T_ASSETS'), axis=1)
	input_data['LAST_YEAR_ASSETS'] = input_data.apply(lambda x: assign_before(x, 'A', input_data, 'T_ASSETS'), axis=1)
	input_data.fillna(0, inplace=True)
	return input_data

def assign_before(x, select='A', input_data=None, attribute=None):
	year = int(x.END_DATE_STR[0][0:4])
	month = int(x.END_DATE_STR[0][5:7])
	day = int(x.END_DATE_STR[0][8:10])
	symbol = x.TICKER_SYMBOL
	end_date = x.END_DATE
	if select == 'A':
		if month < 10:
			tmp = ''+str(year-1) + '-0' + str(month) + '-' + str(day)
		else :
			tmp = ''+str(year-1) + '-' + str(month) + '-' + str(day)
	elif select == 'Q1':
		if month == 3:
			tmp = ''+str(year-1) + '-' + str(12) + '-' + str(31)
		elif month == 6:
			tmp = ''+str(year) + '-0' + str(3) + '-' + str(31)
		elif month == 9:
			tmp = ''+str(year) + '-0' + str(6) + '-' + str(30)
		else:
			tmp = ''+str(year) + '-0' + str(9) + '-' + str(30)
	result = input_data[(input_data.TICKER_SYMBOL == symbol) & (input_data.END_DATE == tmp)] 
	if result.size == 0:
		return 0
	return result[attribute].values[0]


