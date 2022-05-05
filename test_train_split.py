import splitfolders

def test_train_split(input_folder, output_folder):
    splitfolders.ratio(input_folder, output=output_folder,
                       seed=1337, ratio=(.8,.2), group_prefix=None, move=False)

if __name__ == '__main__':

    input_dir = "./data/processed_data"
    output_dir = "./data/split_data"

    test_train_split(input_dir, output_dir)