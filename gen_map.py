from dataprocess.dict_pkl import dictionary_2_pkl_file
from dataprocess.txt_files_2_dict import dictionary_from_txt_files

root_dir = r'C:\Users\Administrator\AnacondaProjects\dataset\wmt14-de-en'
w2i, i2w = dictionary_from_txt_files(
    root_dir=root_dir, files=['europarl-v7.de-en.en']
    )
print(len(w2i))
dictionary_2_pkl_file(w2i, 'data/en_w2i_map')
dictionary_2_pkl_file(i2w, 'data/en_i2w_map')
