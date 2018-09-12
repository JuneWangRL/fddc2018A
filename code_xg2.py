# -*- coding: utf-8 -*-
import pandas as pd 

def market_data_pre(x):
    set_25 = [600816, 600901]
    set_26 =[600291, 601318, 601336, 601601, 601628]
    set_27 = [166, 686, 712, 728, 750, 776, 783, 987, 2500, 2600, 2673, 2736, 2797, 2926, 600030,
             600109, 600369, 600837, 600909, 600958, 600999, 601066, 601099, 601108, 601198,
             601211, 601375, 601377, 601555, 601688, 601788, 601878, 601881, 601901, 601990]
    
    a = -1
    if (x.TICKER_SYMBOL in set_26):
        a=26
    elif (x.TICKER_SYMBOL in set_27):
        a=27
    elif (x.TICKER_SYMBOL in set_25):
        a=25
    elif (x.TYPE_NAME_CN == '非银金融') or (x.TYPE_NAME_CN == '金融服务'):
        a=0
    elif (x.TYPE_NAME_CN == '房地产'):
        a=1
    elif (x.TYPE_NAME_CN == '医药生物'):
        a=2
    elif (x.TYPE_NAME_CN == '公用事业'):
        a=3
    elif (x.TYPE_NAME_CN == '综合'):
        a=4
    elif (x.TYPE_NAME_CN == '休闲服务'):
        a=5
    elif (x.TYPE_NAME_CN == '机械设备') or (x.TYPE_NAME_CN == '电气设备'):
        a=6
    elif (x.TYPE_NAME_CN == '建筑装饰') or (x.TYPE_NAME_CN == '建筑材料') or (x.TYPE_NAME_CN == '建筑建材'):
        a=7
    elif (x.TYPE_NAME_CN == '商业贸易'):
        a=8
    elif (x.TYPE_NAME_CN == '家用电器'):
        a=9
    elif (x.TYPE_NAME_CN == '汽车') or (x.TYPE_NAME_CN == '交运设备'):
        a=10
    elif (x.TYPE_NAME_CN == '纺织服装'):
        a=11
    elif (x.TYPE_NAME_CN == '食品饮料'):
        a=12
    elif (x.TYPE_NAME_CN == '电子'):
        a=13
    elif (x.TYPE_NAME_CN == '计算机'):
        a=14
    elif (x.TYPE_NAME_CN == '信息设备'):
        a=15
    elif (x.TYPE_NAME_CN == '交通运输'):
        a=16
    elif (x.TYPE_NAME_CN == '轻工制造'):
        a=17
    elif (x.TYPE_NAME_CN == '通信'):
        a=18
    elif (x.TYPE_NAME_CN == '农林牧渔'):
        a=19
    elif (x.TYPE_NAME_CN == '化工'):
        a=20
    elif (x.TYPE_NAME_CN == '传媒'):
        a=21
    elif (x.TYPE_NAME_CN == '信息服务'):
        a=22
    elif (x.TYPE_NAME_CN == '有色金属') or (x.TYPE_NAME_CN == '钢铁') or (x.TYPE_NAME_CN == '采掘'):
        a=23
    elif (x.TYPE_NAME_CN == '国防军工'):
        a=24
    elif (x.TYPE_NAME_CN == '银行'):
        a=25
    else: ## 证券
        print('Error')
    return a

