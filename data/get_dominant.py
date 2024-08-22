import rqdatac

rqdatac.init()
dominant = rqdatac.futures.get_dominant("CU", start_date="20100101", end_date="20240809",rule=0)
dominant.to_excel("data/raw_dominant_CU.xlsx")