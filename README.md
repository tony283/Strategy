# 文件层级结构
|  文件夹   |  说明  |
|  ----  | ----  |
| back  | 所有的回测曲线都会生成在back文件夹下，命名格式为Back_${context.name}.xlsx，所有的对账单会放在back/trade/下，命名格式为Trade${context.name}.xlsx |
| data  | 所有的期货历史数据都放在此文件夹下，命名格式为${future_type}_daily.xlsx，每个xlsx都至少包含了以下columns：["close","prev_close","multiplier","date"] |
| Report| 通过back文件夹中的merge_plot.py可以将back中同策略文件夹中不同参数的回测结果形成统计报告保存在Report中|
| Strategy | 所有的策略都放在此文件夹下，具体写法参考里面的sample.py，回测框架的核心BackTestEngine.py放在Strategy/utils下面|
