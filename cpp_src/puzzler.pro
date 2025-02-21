QT += core gui widgets

CONFIG += c++20

SOURCES += \
    main.cpp \
    mainWindow.cpp

HEADERS += \
    mainWindow.h

FORMS += \
    mainWindow.ui

RESOURCES += \
    resources.qrc

PRECOMPILED_HEADER = precompiled_headers.h

win32:VERSION = 0.0.1.0 # major.minor.patch.build
else:VERSION = 0.0.1    # major.minor.patch

CONFIG += embed_translations

# add translations of qtbase as resource
qtbase_qm_files.files = $$files($$[QT_INSTALL_TRANSLATIONS]/qtbase_*.qm)
qtbase_qm_files.base = $$[QT_INSTALL_TRANSLATIONS]
qtbase_qm_files.prefix = i18n
RESOURCES += qtbase_qm_files

QMAKE_CXXFLAGS += -Wno-deprecated-enum-enum-conversion

INCLUDEPATH += C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\include

LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_calib3d4100.dll.a
LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_core4100.dll.a
#LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_dnn4100.dll.a
LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_features2d4100.dll.a
#LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_flann4100.dll.a
#LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_gapi4100.dll.a
LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_highgui4100.dll.a
LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_imgcodecs4100.dll.a
LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_imgproc4100.dll.a
#LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_ml4100.dll.a
#LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_objdetect4100.dll.a
#LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_photo4100.dll.a
#LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_stitching4100.dll.a
#LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_video4100.dll.a
#LIBS +=  C:\Users\daniel\Downloads\opencv-4.10.0\build-opencv-4.10.0-Desktop_Qt_6_8_0_MinGW_64_bit-Release\install\x64\mingw\lib\libopencv_videoio4100.dll.a

