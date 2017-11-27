# Install TesorFlow and Keras with Ananconda on Windows

### Download and install the Anaconda. 
Donwload and install the latest version of Anaconda from [official website](https://www.anaconda.com/download/). 

[Create a conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) by invoking the following command (here it is named my_tf):
```
C:> conda create -n my_tf python=3.5 
```
Activate the conda environment by issuing the following command:
```
C:> activate my_tf
 (tensorflow)C:>  # Your prompt should change
```

### 2 Install TensorFlow through Anaconda
After successfully installed Anaconda, navigate to [this page](https://www.tensorflow.org/install/install_windows#installing_with_anaconda) then follow the instructions to install the TensorFlow on windows through the Anaconda. 

Please make sure that the following will be operated under the created Ancaonda envirionment, so if you start freshly with the Anaconda Prompt, issue the following first (repalce env_name with the name of desired envirionment)
```
activate env_name
```
If you start from windows commamd line, i.e, run cmd by Ctrl+R. Navigate to Suppose the directory C:\Users\user-name\Anaconda3\Scripts, then run
```
activate my_tf
```
Replace "user-name" in the path with the actual user name.

To install the CPU-only version of TensorFlow, enter the following command:
```
(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow 
```
To install the GPU version of TensorFlow, enter the following command (on a single line):
```
(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow-gpu 
```

### 3 Install Keras
Firstly, install a few Python dependencies (under the same conda environment where tensorflow is installed):
```
pip install h5py
pip install pillow
```
Then, install keras:
```
pip install keras
```
