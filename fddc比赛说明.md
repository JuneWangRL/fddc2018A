### 预测A股2018年第2季度的营收

#### 模型：xgb、lgb、gbdt、arima

#### 一. 构建特征工程：

给了123多维的数据特征，但是很多数据都不是特征，所以选取数据特征的时候要考虑选取的特征是否跟最后需预测的值相关。

##### xgb、lgb、gbdt选取的数据特征：

@1.上一期的营收；@2.上一年同期的数据；@3.上一期的资产；@4.上一年同期的资产

##### arima选择的数据特征：

前三年的营收数据，时间过长市场变动太大

```
#arima关键代码
def station_test(ts):    #检验平稳性
    dftest=adfuller(ts,maxlag=10)
    df_p=dftest[1]
    if df_p>=0.05:
        stationarity=False
    elif df_p<0.05:
        stationarity=True
    return stationarity

def ARMA_MODEL(timeseries): #返回预测值
    order=sm.tsa.arma_order_select_ic(timeseries,max_ar=3,max_ma=3,ic='bic')['bic_min_order'] 
    temp_model= sm.tsa.ARMA(timeseries,order).fit()
    pred=temp_model.forecast(1) #返回后面一期
    return pred[0][-1]

def decompose2(timeseries): #差分
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
```

##### 缺失值处理：插值法

```
data.interpolate(method='linear')
datatest1.fillna(method='ffill',inplace=True)
datatest1.fillna(method='bfill',inplace=True)
```

##### GBDT

```
#主要模型，调参数
model = GradientBoostingRegressor(learning_rate=0.1, max_depth=j,min_samples_split=k, n_estimators=1500, loss='ls')
model.fit(sub_type_data_in, np.array(sub_type_data_out).ravel())

```

学习速率：0.05或0.1，树的个数越大学习速率越小，超过2000写成0.05

树的最大深度：3-10，树过深会过拟合，树过浅就欠拟合

节点个数：100-250，控制终止条件

树的个数：1000-2000

说明：最后看验证集和测试集的得分，若得分相差较大，说明过拟合。

若缺少测试集，则利用cross validation的方法

##### xgboost

```
model=XGBRegressor(learning_rate=0.1,n_estimators=2200,max_depth=3,min_child_weight=7,gamma=0,subsample=0.8,colsample_bytree=1,reg_alpha=0.005, objective='reg:linear')
model.fit(sub_type_data_in, np.array(sub_type_data_out).ravel())
```

##### lightgbm

```
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
	gbm = lgb.train(params,lgb_train,num_boost_round=500,valid_sets=lgb_eval,early_stopping_rounds=50)
```

