from dataprocess.dict_pkl import dictionary_2_pkl_file
from dataprocess.txt_files_2_dict import dictionary_from_txt_files
import sys
import os

def gen_map(root_dir, file):
    w2i, i2w = dictionary_from_txt_files(
        root_dir=root_dir, files=[file]
        )
    dictionary_2_pkl_file(w2i, os.path.join(root_dir, '{}_w2i_map'.format(file)))
    dictionary_2_pkl_file(i2w, os.path.join(root_dir, '{}_i2w_map'.format(file)))


if __name__ == '__main__':
    argv = sys.argv
    gen_map(argv[0], argv[1])