from contracts import *





handle_contract("CU","20120102","20240827",
                DeliveryMonthCondition(days=10,working_day=True),
                VolumeCondition(days=3,max_volume=1000),###条件可以自由添加
                )