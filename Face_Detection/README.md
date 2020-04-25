Benchmark of Publicly Available Face Model on WIDER Dataset

In this repository, we provide several face detection modules and models which you can use for your own application. We also provided script to benchmark performance of each techniques on WIDER Face dataset

The available face detection techniques in this repositories are:

1. OpenCV Haar Cascades Classifier
2. DLib HOG
3. DLib CNN
4. Multi-task Cascaded CNN (Tensorflow)
5. Mobilenet-SSD Face Detector (Tensorflow)

WIDER Face dataset
You can download WIDER Face Dataset in this link. For benchmarking purpose, we only use train and validation split of the dataset because only those split which have face bounding boxes ground truth information. In order to use the benchmarking script, you should put the data under folder dataset like this structure :

+-- dataset
|   +-- WIDER_train
|   +-- WIDER_val
|   +-- wider_face_split

Link: https://github.com/nodefluxio/face-detector-benchmark

Another link:

1. https://github.com/rosaj/face_detection

2. https://github.com/timesler/facenet-pytorch

3. https://github.com/deepinsight/insightface
