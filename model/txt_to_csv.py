import csv
import zlib

with open('C:/final_project/model/test_data/test_box/test5_annotation.txt', 'r', encoding='utf-8') as in_file:
    lines = in_file.read().splitlines()
    stripped = [line.replace(","," ").split() for line in lines]
    grouped = zip(*[stripped]*1)

    with open('C:/final_project/model/test_data/test_box/test5.csv', 'w', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('A', 'B', 'C', 'D', 'E', 'F'))

        for group in grouped:
            writer.writerows(group)