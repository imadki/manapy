
# Download the data from https://drive.google.com/file/d/1mpG9yru_4hid40ovjHQZ0cznAGtOwH0z/view?usp=sharing
# unzip it

mkdir -p build
cd build
cmake ..
make

# ./ginkgo_solver reference gmres small_data