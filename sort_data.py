#Cite: https://stackoverflow.com/questions/23178039/i-need-to-write-a-python-script-to-sort-pictures-how-would-i-do-this

import os
import os.path
import shutil


def sort_data(input_directory):

    videos = [v for v in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, v))]

    for vid in videos:
        category = vid.split('_')[0]
        folder_name = category

        new_path = os.path.join(input_directory,folder_name)

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        old_video_path = os.path.join(input_directory, vid)
        new_video_path = os.path.join(new_path, vid)
        shutil.move(old_video_path, new_video_path)

if __name__ == '__main__':

    sort_data("./data/input_folder/")

