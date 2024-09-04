import requests
import pandas as pd
from urllib import request
import json
from urllib.parse import urlencode
def get_current_price(future_type):
    spbz = f"{future_type}0"
    headers = {'referer': 'http://finance.sina.com.cn'}
    resp = requests.get('http://hq.sinajs.cn/list=nf_' + spbz, headers=headers, timeout=6)  # 请求网页+a[i6]
    content1 = resp.text
    content1 = content1.split(",")
    return float(content1[6])

def get_all_future_price():
    typelist = ['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
          'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
          'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
          'SI', 'SH', 'PX', 'BR', 'AO']
    df = pd.DataFrame(index=typelist,columns=["close"])
    for i in typelist:
        df.loc[i,"close"]=get_current_price(i)
    print(df)
    return df

# get_all_future_price()
def get_ftures_k_history(secid: str, beg: str = '19000101', end: str = '20500101', klt: int = 101, fqt: int = 1) -> pd.DataFrame:
    '''
    获取k线数据

    Parameters
    ----------
    secid : 根据 get_futures_base_info 函数获取
    获取4个交易所期货数据，取 secid 列来获取 secid
    beg : 开始日期 例如 20200101
    end : 结束日期 例如 20200201
    klt : k线间距 默认为 101 即日k
            klt : 1 1 分钟
            klt : 5 5 分钟
            klt : 101 日
            klt : 102 周
    fqt: 复权方式
            不复权 : 0
            前复权 : 1
            后复权 : 2 

    Return
    ------
    DateFrame : 包含期货k线数据

    '''
    EastmoneyHeaders = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; Touch; rv:11.0) like Gecko',
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Referer': 'http://quote.eastmoney.com/center/gridlist.html',
    }
    EastmoneyKlines = {
        'f51': '日期',
        'f52': '开盘',
        'f53': '收盘',
        'f54': '最高',
        'f55': '最低',
        'f56': '成交量',
        'f57': '成交额',
        'f58': '振幅',
        'f59': '涨跌幅',
        'f60': '涨跌额',
        'f61': '换手率',


    }
    fields = list(EastmoneyKlines.keys())
    columns = list(EastmoneyKlines.values())
    fields2 = ",".join(fields)

    params = (
        ('fields1', 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13'),
        ('fields2', fields2),
        ('beg', beg),
        ('end', end),
        ('rtntype', '6'),
        ('secid', secid),
        ('klt', f'{klt}'),
        ('fqt', f'{fqt}'),
    )
    base_url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    url = base_url+'?'+urlencode(params)
    json_response = requests.get(
        url, headers=EastmoneyHeaders).json()

    data = json_response['data']
    if data is None:
        print(secid, '无数据')
        return None
    # code = data['code']
    # name
    # name = data['name']
    klines = data['klines']

    rows = []
    for _kline in klines:

        kline = _kline.split(',')
        rows.append(kline)

    df = pd.DataFrame(rows, columns=columns)

    return df
get_ftures_k_history("RB0")