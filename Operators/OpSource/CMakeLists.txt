INCLUDE_DIRECTORIES(
    /usr/local/lib/python3.5/dist-packages/tensorflow
    /usr/local/lib/python3.5/dist-packages/tensorflow/include
)

SET(LIBRARY_OUTPUT_PATH ../../lib)
ADD_LIBRARY(binarizeOp SHARED Binarize.cpp)

TARGET_LINK_LIBRARIES( binarizeOp /usr/local/lib/python3.5/dist-packages/tensorflow/libtensorflow_framework.so )