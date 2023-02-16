echo "Configuring and building ORB_SLAM2 ..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..

echo
echo "Test Python test.py"
echo
python test.py

echo
echo "Test C++ opencv_vs_slam.cc "
echo
./build/opencv_vs_slam

