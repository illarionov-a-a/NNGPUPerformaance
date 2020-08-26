NODENAME=$1
USE_GPU=$2

BOOST_INCLUDE=./boost/include
TF_INCLUDE=./libtensorflow/include/$USE_GPU
TF_LIB=./libtensorflow/lib/$USE_GPU
NJOBS=1


case $NODENAME in
"CLUSTER2" | "Cluster2" )
C_COMP=/opt/rh/devtoolset-7/root/bin/gcc
CXX_COMP=/opt/rh/devtoolset-7/root/bin/g++
#BOOST_INCLUDE=/share/COMMON/MDRUNS/ARBALEST_BOOST_INCLUDE
;;
"titanium" )
C_COMP=gcc
CXX_COMP=g++
#BOOST_INCLUDE=/etc/arbalest/ARBALEST_BOOST_INCLUDE
;;
esac

#case $USEGPU in
#"gpu" )
#TF_LIB=./libtensorflow/lib/gpu
#;;
#esac

CMAKE_COMP_STR="-DUSE_GPU=$USE_GPU  -DCMAKE_C_COMPILER=$C_COMP -DCMAKE_CXX_COMPILER=$CXX_COMP  -DTF_LIB=$TF_LIB -DTF_INCLUDE=$TF_INCLUDE -DBOOST_INCLUDE=$BOOST_INCLUDE "

#make clean 
#rm Makefile
find -iname '*cmake*' -not -name CMakeLists.txt -exec rm -rf {} \+
cmake $CMAKE_COMP_STR
make VERBOSE=1  -j $NJOBS

