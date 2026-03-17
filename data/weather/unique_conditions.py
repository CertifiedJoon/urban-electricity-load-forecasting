import csv
uniq = set()
with open('combined_weather.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        uniq.add(row[2])
print(len(uniq))