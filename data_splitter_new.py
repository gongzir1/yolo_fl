import os
import shutil
import random
import glob

def mv_file(img, num) :

    img_path_list = os.listdir(img)

    #img_path_list = glob.glob(f"{img}/*." + img_ext)

    random.shuffle(img_path_list)

    if num > len(img_path_list):
        print('Length need to be small than：', len(img_path_list))
        exit()
    num_file = int(len(img_path_list) / num)  + 1
    cnt = 0
    for n in range(1, num_file + 1):  #create files
        new_file = os.path.join(img + '_' + str(n))
        if os.path.exists(new_file + '_' + str(cnt)):
            print('The file already exists, please solve the conflict', new_file)
            exit()
        print('creat new files：', new_file)
        os.mkdir(new_file)
        list_n = img_path_list[num * cnt:num * (cnt + 1)]
        for m in list_n:
            old_img_path = os.path.join(img, m)
            #old_txt_path = old_img_path[0: len(old_img_path) - 3] + "txt"
            new_img_path = os.path.join(new_file, m)
            #new_txt_path = new_img_path[0: len(new_img_path) - 3] + "txt"

            shutil.copy(old_img_path, new_img_path)
            #shutil.copy(old_txt_path, new_txt_path)
        cnt = cnt + 1
    print('============task OK!===========')


if __name__ == "__main__":
    path = '/data/datasets/meat'
    mv_file(path, 864)