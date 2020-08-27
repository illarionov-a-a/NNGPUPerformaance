USE_GPU=$1
NODENAME=$2

if [ "$USE_GPU" != "cpu" ] && [ "$USE_GPU" != "gpu" ]; then
  echo "Value of first argument must be 'gpu' or 'cpu' "
  exit 1
fi

BOOST_INCLUDE=./boost/include
TF_INCLUDE=./libtensorflow-$USE_GPU-linux-x86_64-1.15.0/include
TF_LIB=./libtensorflow-$USE_GPU-linux-x86_64-1.15.0/lib
C_COMP=gcc
CXX_COMP=g++
NJOBS=1

case $NODENAME in
"CLUSTER2" | "Cluster2" )
C_COMP=/opt/rh/devtoolset-7/root/bin/gcc
CXX_COMP=/opt/rh/devtoolset-7/root/bin/g++
;;
"titanium" | "default" )
C_COMP=gcc
CXX_COMP=g++
;;
esac

CMAKE_COMP_STR="-DUSE_GPU=$USE_GPU  -DCMAKE_C_COMPILER=$C_COMP -DCMAKE_CXX_COMPILER=$CXX_COMP  -DTF_LIB=$TF_LIB -DTF_INCLUDE=$TF_INCLUDE -DBOOST_INCLUDE=$BOOST_INCLUDE "

make clean 
rm Makefile
find -iname '*cmake*' -not -name CMakeLists.txt -exec rm -rf {} \+
cmake $CMAKE_COMP_STR
make VERBOSE=1  -j $NJOBS

