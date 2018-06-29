INCLUDE_DIRECTORIES(
    /usr/local/lib/python3.5/dist-packages/tensorflow
    /usr/local/lib/python3.5/dist-packages/tensorflow/include
)

SET(LIBRARY_OUTPUT_PATH ../../lib)
ADD_LIBRARY(testop SHARED testop.cpp)

TARGET_LINK_LIBRARIES( testop /usr/local/lib/python3.5/dist-packages/tensorflow/libtensorflow_framework.so )