# CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(TrajectoryFitter)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# 查找依赖库
# ------------------------------------------------------------------

# 查找 Eigen 库
find_package(Eigen3 REQUIRED)

# 查找 Ceres 库
find_package(Ceres REQUIRED)

# 关键部分：明确指定我们自定义编译的 OpenCV 版本的路径
# !!! 将 'zyanju' 替换为你的实际用户名 !!!
set(OpenCV_DIR "/home/zyanju/opencv_custom/4.5.4/lib/cmake/opencv4")

# 查找 OpenCV 库 (CMake 会优先使用上面指定的路径)
find_package(OpenCV REQUIRED COMPONENTS opencv_core opencv_imgproc opencv_highgui opencv_tracking)

# ------------------------------------------------------------------
# 添加可执行文件
# ------------------------------------------------------------------

# 将你的 C++ 文件名填在这里
add_executable(fit_trajectory main.cpp)

# ------------------------------------------------------------------
# 添加包含目录 (好习惯)
# ------------------------------------------------------------------
target_include_directories(fit_trajectory PRIVATE 
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# ------------------------------------------------------------------
# 链接库
# ------------------------------------------------------------------
target_link_libraries(fit_trajectory PRIVATE 
    # 使用 Ceres 定义的传统变量来链接
    ${CERES_LIBRARIES} 
    # 使用 OpenCV 定义的变量来链接
    ${OpenCV_LIBS}
)

# ------------------------------------------------------------------
# 打印信息用于调试
# ------------------------------------------------------------------
message(STATUS "Found Eigen3 version: ${EIGEN3_VERSION_STRING}")
message(STATUS "Found Ceres version: ${CERES_VERSION}")
message(STATUS "Found OpenCV version: ${OpenCV_VERSION} in ${OpenCV_DIR}")