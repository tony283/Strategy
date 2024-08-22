import numpy as np

def try_set_value(a:dict,key,value):
    """_summary_

    Args:
        a (dict): _description_
        key (_type_): _description_
        value (_type_): [amount,price,margin]
    """
    assert value[0]>=0
    if (key in a.keys()):
        originial_hold =a[key][0]
        a[key][0]+=value[0]
        
        a[key][1]=(value[0]*value[1]+originial_hold*a[key][1])//a[key][0]
        a[key][2]+=value[2]
    else:
        a[key]=np.array(value)
def try_sell_value(a:dict,key,value,direction):
    """_summary_

    Args:
        a (dict): _description_
        key (_type_): _description_
        value (_type_): [amount,price]
        direction (_type_): _description_
    """
    assert value[0]>=0
    assert direction=="long" or direction=="short"
    assert value[0]<=a[key][0]
    earn=0
    if key+"_"+direction in a.keys():
        originial_hold =a[key][0]
        signal = 1 if direction=="long" else -1
        earn=  signal*value[0]*(value[1]-a[key][1])
        a[key][2] -=earn#amount*close_price
        a[key][0] -=value[0]
    return earn
        