# FaceQnet
FaceQnet: Quality Assessment for Face Recognition based on Deep Learning

This repository contains the DNN FaceQnet presented in the paper: <a href="https://arxiv.org/abs/1904.01740" rel="nofollow">"FaceQnet: Quality Assessment for Face Recognition based on Deep Learning"</a>.

FaceQnet is a No-Reference, end-to-end Quality Assessment (QA) system for face recognition based on deep learning. 
The system consists of a Convolutional Neural Network that is able to predict the suitability of a specific input image for face recognition purposes. 
The training of FaceQnet is done using the VGGFace2 database.

-- Configuring environment in Windows:

1) Installing Conda: https://conda.io/projects/conda/en/latest/user-guide/install/windows.html

  Update Conda in the default environment:

    conda update conda
    conda upgrade --all

  Create a new environment:

    conda create -n [env-name]

  Activate the environment:

    conda activate [env-name]

2) Installing dependencies in your environment:

  Install Tensorflow and all its dependencies: 
    
    pip install tensorflow
    
  Install Keras:
  
    pip install keras
    
  Install OpenCV:

    conda install -c conda-forge opencv
  
 3) If you want to use a CUDA compatible GPU for faster predictions:
  
   You will need CUDA and the Nvidia drivers installed in your computer: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/
  
   Then, install the GPU version of Tensorflow:
    
    pip install tensorflow-gpu
  
-- Using FaceQnet for predicting scores:

  1) Download or clone the repository. 
  2) Due to the size of the video example, please download the FaceQnet pretrained model <a href="https://github.com/uam-biometrics/FaceQnet/releases/download/v1.1/FaceQnet.h5" rel="nofollow">here</a> (.h5 file) and place it in the /src folder.
  3) Edit and run the FaceQNet_obtainscores_Keras.py script.
     - You will need to change the folder from which the script will try to charge the face images. It is src/Samples_cropped by default. 
     - The best results will be obtained when the input images have been cropped just to the zone of the detected face. In our experiments we have used the MTCNN face detector from <a href="https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html" rel="nofollow">here</a>, but other detector can be used.
     - FaceQnet will ouput a quality score for each input image. All the scores will are saved in a .txt file into the src folder. This file contain each filename with its associated quality metric.





