import json
import csv


with open('result.json') as json_file:
    data = json.load(json_file)


print(data[0]['entities'])
# for i in range(0,1):
#     print(data[i])

# data_file = open('result.csv', 'w')

# csv_writer = csv.writer(data_file)

# header = ['id', 'title', 'entity', 'timeline', 'url']

# for x in data:
#     row = []
#     row.push(x['id'])
#     row.push(x['title'])
#     row.push(x['url'])
    

