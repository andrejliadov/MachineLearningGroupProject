import glob
from os import mkdir, listdir
from pathlib import Path
from shutil import copy

#Parse words in the text files into their catagories
def parse_category(category):
    text_data = ""
    text_data_list = []
    mkdir(output_dir + category)
    total_duplicates = 0



    for filename in glob.iglob(root_dir + category + '/*.txt', recursive=True):
        file_count = 0
        file_id = filename.split(sep="/")[-1]
        # print(file_id)
        for other_filename in Path(root_dir).rglob('**/*.txt'):
            other_file_id = other_filename.name.split(sep="/")[-1]
            if file_id == other_file_id: file_count += 1;
        # print(file_count)
        if file_count == 1:
            # print(filename)
            copy(filename, output_dir + category + '/')
        else:
            total_duplicates += 1;

    print(str(total_duplicates) + ' removed from ' + category )
    return total_duplicates


output_dir = 'clean-10-categories-data/'
output_modifier = 'clean-'
root_dir = '10-categories-data/'

mkdir(output_dir)
categories = listdir(root_dir)
print(categories)
for category in categories:
    parse_category(category)