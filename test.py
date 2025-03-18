import torch


def find_index_from_right(lst, value):
    reversed_index = lst[::-1].index(value[::-1])
    return len(lst) - 1 - reversed_index

filename = "I-210618_I01001_W01_F0001.jpg"

index = find_index_from_right(filename, '_F')

new_filename = filename[:index]
print(new_filename)