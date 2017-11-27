In this folder, we put files for installing the Tensorflow, Keras, OpenCV, and dlib based on the [Anaconda](https://www.anaconda.com/download/). Below we give a brief summary of the essential steps:

### 1 Download and install the Anaconda. 

After successfully installed Anaconda, navigate to [this page](https://www.tensorflow.org/install/install_windows#installing_with_anaconda) then follow the instructions to install the TensorFlow on windows through the Anaconda. Be advised that, one should operate in the Anaconda Prompt instead of windows command prompt. Just find the Anaconda Prompt in the windows start menu, and give it a run. 


### 2 Install TensorFlow through Anaconda
2.1 Create a conda environment named tensorflow by invoking the following command:
```
C:> conda create -n tensorflow python=3.5 
```
2.2 Activate the conda environment by issuing the following command:
```
C:> activate tensorflow
 (tensorflow)C:>  # Your prompt should change
```
If you start from windows commamd line, i.e, run cmd by Ctrl+R. Suppose the current directory is C:\Users\user-name\, and the conda environment name is "env-name" then issue the following command to navigate to the Scripts folder:
```
cd Anaconda3/Scripts/
```
then run
```
activate env-name
```
to enter the conda environment "env-name"

2.3 Issue the appropriate command to install TensorFlow inside your conda environment. To install the CPU-only version of TensorFlow, enter the following command:
```
(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow 
```
To install the GPU version of TensorFlow, enter the following command (on a single line):
```
(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow-gpu 
```
