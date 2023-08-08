from pymongo import MongoClient
import datetime
import pprint

client = MongoClient(host='10.44.237.51', port=27017)
db = client.mydb

post = {"Target":"Ti-B rod 투입 현황",
        "x_col":"Timestamp",
        "y_col": "DC_3 ROD_PV_TiBorSpeed",
        "plot":"line_raw_multi",
        "y_min": 30, "y_max":60
}
print(post)

# Collection 리스트 조회
print(db.list_collection_names())