###### Generate prediction data to evaluate pre-trained model on WIDER test dataset (easy/med/hard)
'''
For example, if the directory of a testing image is "./0--Parade/0_Parade_marchingband_1_5.jpg", the detection result should be writtern in the text file in "./0--Parade/0_Parade_marchingband_1_5.txt". The detection output is expected in the follwing format:
...
< image name i >
< number of faces in this image = im >
< face i1 >
< face i2 >
...
< face im >
'''

import cv2
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
import os

eval_root='/content/drive/My Drive/mtcnn-pytorch/data_set/face_detection/WIDERFACE/WIDER_test/images'
eval_list=os.listdir(eval_root)
predict_root='WFPred'

for dire in eval_list:
    if not os.path.exists(os.path.join(predict_root, dire)):
           os.makedirs(os.path.join(predict_root, dire))
    image_list= os.listdir(os.path.join(eval_root, dire))
    img_base_dir =  '/content/drive/My Drive/mtcnn-pytorch/data_set/face_detection/WIDERFACE/WIDER_test/images/'
    for imageFile in image_list:
        lst_write_ret = []
        img_full_name = img_base_dir + dire + '/' + imageFile
        pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt", r_model_path="./original_model/rnet_epoch.pt", o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
        mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)
        img = cv2.imread(img_full_name)
        img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred, landmarks = mtcnn_detector.detect_face(img)
        lst_write_ret.append(imageFile)
        lst_write_ret.append(str(len(pred)))
        if len(pred) > 0:
        # append each face rectangle x y w h score
          for face_rect in pred:
        # append face rectangle x, y, w, h score
              s_rect = " ".join(str(item) for item in face_rect)
              lst_write_ret.append(s_rect)
        print(lst_write_ret)
        with open(os.path.join(predict_root, dire, imageFile.rsplit( ".", 1 )[ 0 ]+'.txt'), 'w', encoding='utf8') as fw:
            for line in lst_write_ret:
                  fw.write("%s\n" % line)
