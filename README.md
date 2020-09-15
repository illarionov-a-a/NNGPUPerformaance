# NNGPUPerformaance
To build and run project 
1 Please download tensorflow library for cpu/gpu linnux/windows:

	https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz
    https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz
    
    https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.15.0.zip
    https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.15.0.zip
Unpack archives and put them to project directory.

2 Download boost https://www.boost.org/users/download/ unpuck it and put boost include file to 
project_dir/boost/include/boost/. You don't need to build boost to run this project.

3 Compile project for linux run in project directory : 
./c_make-run.sh default cpu   
or 
./c_make-run.sh default gpu   
For windows use VS 20019 to build project.

