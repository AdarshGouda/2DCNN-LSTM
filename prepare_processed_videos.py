import os
import os.path
from tqdm import tqdm
import cv2
from mask import masking_operation


def rescale_frame(frame, target_shape):
    width = int(target_shape)
    height = int(target_shape)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def process_video(source_directory, output_dir, target_shape=160):

    root_ = os.getcwd()
    output_abspth = os.path.abspath(output_dir)

    if not os.path.exists(output_abspth):
        os.makedirs(output_abspth)

    source_directory = os.path.abspath(source_directory)

    os.chdir(source_directory)
    sub_folders = os.listdir(os.getcwd())

    for folder in tqdm(sub_folders, unit="actions", ascii=True):
        folder_path = os.path.join(source_directory, folder)
        os.chdir(folder_path)

        output_file_path = os.path.join(output_abspth, folder)
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)

        files = os.listdir(os.getcwd())

        for file in tqdm(files, unit="actions", ascii=False):
            name = os.path.abspath(file)
            class_ = int(file.split('_')[0])

            frames = []
            count = 0

            video = cv2.VideoCapture(name)

            if video.isOpened():
                ret, frame = video.read()
                rescaled_frame = rescale_frame(frame, target_shape=target_shape)
                (h, w) = rescaled_frame.shape[:2]
                #imageWidth = int(video.get(3))
                #imageHeight = int(video.get(4))
                fps = video.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                os.chdir(output_file_path)

                framename = os.path.splitext(file)[0]
                framename = framename + ".mp4"

                out = cv2.VideoWriter(framename, fourcc, fps, (w, h), True)




            while (video.isOpened()):
                ret, frame = video.read()
                if ret == True:
                    frame = masking_operation(frame)
                    rescaled_frame = rescale_frame(frame, target_shape=target_shape)
                    frame = cv2.cvtColor(rescaled_frame, cv2.COLOR_BGR2GRAY)
                    #frame = cv2.resize(frame, (target_shape,target_shape))
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    out.write(frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break

            os.chdir(folder_path)
            video.release()
            out.release()
            cv2.destroyAllWindows()

    os.chdir(root_)


if __name__ == '__main__':

    process_video(source_directory="./data/input_folder/", output_dir="./data/processed_folder/",target_shape = 160)





