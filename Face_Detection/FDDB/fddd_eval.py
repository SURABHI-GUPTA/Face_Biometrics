##### To generate prediction result on FDDB Dataset in the required format #####
##### Same code used to generate fddb-out.txt files #####

import cv2
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector

def get_img_relative_path():
    """
    :return: ['2002/08/11/big/img_344', '2002/08/02/big/img_473', ......]
    """
    f_name = '/home/surabhi/Desktop/FDDB/all_img_files.txt'                          
    lst_name = open(f_name).read().split('\n')
    while("" in lst_name) : 
    	lst_name.remove("")

    return lst_name

def write_lines_to_txt(lst):
    # lst = ['line1', 'line2', 'line3']
    f_path = 'fddb_rect_eval.txt'
    with open(f_path, 'w') as fp:

        for line in lst:
            fp.write("%s\n" % line)

# For example use opencv to face detection
def detect_face_lst(img):
    """
    :param img: opencv image 
    :return: face rectangles [[x, y, w, h], ..........]
    """
    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt", r_model_path="./original_model/rnet_epoch.pt", o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    bboxs, landmarks = mtcnn_detector.detect_face(img)

    return bboxs


def generate_fddb_ret():
    # The directory from which we get the test images from FDDB
    img_base_dir = '/home/surabhi/Desktop/FDDB/originalPics/'

    # All the images relative path, like '['2002/08/11/big/img_344', '2002/08/02/big/img_473', ......]'
    lst_img_name = get_img_relative_path()

    # Store detect result, like:
    # ['2002/08/11/big/img_344', '1', '10 10 50 50 1', .............]
    lst_write2_fddb_ret = []

    try:
        for img_name in lst_img_name:
            img_full_name = img_base_dir + img_name + '.jpg'
            img = cv2.imread(img_full_name)

            lst_face_rect = detect_face_lst(img)

            # append img name like '2002/08/11/big/img_344'
            lst_write2_fddb_ret.append(img_name)

            face_num = len(lst_face_rect)
            # append face num, note if no face 0 should be append
            lst_write2_fddb_ret.append(str(face_num))

            if face_num > 0:
                # append each face rectangle x y w h score
                for face_rect in lst_face_rect:
                    # append face rectangle x, y, w, h score
                    # note: opencv hava no confidence so use 1 here
                    s_rect = " ".join(str(item) for item in face_rect)
                    lst_write2_fddb_ret.append(s_rect)

    except Exception as e:
        print('error %s , can not generate complete fddb evaluate file', e)
        return -1

    # Write all the result to txt for FDDB evaluation
    write_lines_to_txt(lst_write2_fddb_ret)


if __name__ == "__main__":
     generate_fddb_ret()
