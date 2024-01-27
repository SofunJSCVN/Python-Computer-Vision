import json


data = {}
data["0"]={}
with open('./storage/data.json', 'w') as f:
    json.dump(data, f)