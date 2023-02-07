echo "Configuring and building ORB_SLAM2 ..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..

#echo
#echo "Test:"
#python3 test.py
