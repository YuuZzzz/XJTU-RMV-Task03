#include <iostream>
#include <vector>
#include <cmath> 
#include <ceres/ceres.h>
#include "glog/logging.h"
#include <opencv2/opencv.hpp>

const std::string VIDEO_FILE_PATH = "/home/zyanju/ceres-project/video.mp4";
const int BLACK_THRESHOLD_V = 30;
const int MIN_CONTOUR_AREA = 20;
const double MIN_CIRCULARITY = 0.80;

struct BallisticModel {
    template <typename T>
    void operator()(const T* params, const T& t, T* predicted_x, T* predicted_y) const {
        const T& x0 = params[0]; const T& y0 = params[1];
        const T& v0x = params[2]; const T& v0y = params[3];
        const T& g = params[4]; const T& k = params[5];
        T exp_kt = ceres::exp(-k * t);
        *predicted_x = x0 + (v0x / k) * (T(1.0) - exp_kt);
        *predicted_y = y0 + ((v0y + g / k) / k) * (T(1.0) - exp_kt) - (g / k) * t;
    }
};

struct BallisticCostFunctor {
    BallisticCostFunctor(double t, double observed_x, double observed_y)
        : t_(t), observed_x_(observed_x), observed_y_(observed_y) {}
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        T predicted_x, predicted_y;
        BallisticModel model;
        model(params, T(t_), &predicted_x, &predicted_y);
        residual[0] = T(observed_x_) - predicted_x;
        residual[1] = T(observed_y_) - predicted_y;
        return true;
    }
private:
    const double t_; const double observed_x_; const double observed_y_;
};

bool extract_trajectory_from_video(
    const std::string& video_path,
    std::vector<double>& t_data,
    std::vector<double>& x_data,
    std::vector<double>& y_data,
    cv::Mat& out_background_frame)
{
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) { /* error handling */ return false; }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 60.0;

    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int frame_count = 0;
    cv::Mat frame;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        
        if (frame_count == 0) { out_background_frame = frame.clone(); }

        //颜色过滤 形态学操作
        cv::Mat hsv_frame, non_black_mask, blue_mask, combined_mask;
        cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);
        cv::inRange(hsv_frame, cv::Scalar(0, 0, BLACK_THRESHOLD_V), cv::Scalar(179, 255, 255), non_black_mask);
        cv::inRange(hsv_frame, cv::Scalar(90, 40, 40), cv::Scalar(140, 255, 255), blue_mask);
        cv::bitwise_and(non_black_mask, blue_mask, combined_mask);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(combined_mask, combined_mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(combined_mask, combined_mask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(combined_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double max_area = 0;
        int best_contour_idx = -1;

        //轮廓筛选
        for (int i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);

            if (area < MIN_CONTOUR_AREA) {
                continue;
            }

            //过滤掉不够圆的轮廓
            double perimeter = cv::arcLength(contours[i], true);
            if (perimeter == 0) {
                continue;
            }
            double circularity = (4 * M_PI * area) / (perimeter * perimeter);
            
            if (circularity < MIN_CIRCULARITY) {
                continue;
            }

            //找到面积最大的
            if (area > max_area) {
                max_area = area;
                best_contour_idx = i;
            }
        }

        //质心定位
        if (best_contour_idx != -1) {
            const auto& best_contour = contours[best_contour_idx];
            cv::Moments M = cv::moments(best_contour);
            if (M.m00 > 0) {
                double center_x = M.m10 / M.m00;
                double center_y = M.m01 / M.m00;

                double current_t = static_cast<double>(frame_count) / fps;
                double current_x_phys = center_x;
                double current_y_phys = frame.rows - center_y;

                t_data.push_back(current_t);
                x_data.push_back(current_x_phys);
                y_data.push_back(current_y_phys);
            }
        }
        
        frame_count++;
        int progress = static_cast<int>(100.0 * frame_count / total_frames);
    }
    std::cout << "\nTrajectory extraction complete. Found " << t_data.size() << " points.\n\n";

    cap.release();
    
    if (t_data.size() < 10) { /* error handling */ return false; }
    
    return true;
}

void visualize_and_save_results(
    const cv::Mat& background,
    const std::vector<double>& t_data,
    const std::vector<double>& x_data,
    const std::vector<double>& y_data,
    const double* const fit_params)
{
    cv::Mat result_image = background.clone();
    int img_height = result_image.rows;

    //实际观测轨迹
    for (size_t i = 0; i < x_data.size(); ++i) {
        // 将y坐标从物理坐标系转换回OpenCV图像坐标系
        cv::Point center(cv::saturate_cast<int>(x_data[i]), 
                         cv::saturate_cast<int>(img_height - y_data[i]));
        cv::circle(result_image, center, 3, cv::Scalar(255, 0, 0), -1);
    }

    //Ceres拟合轨迹
    BallisticModel model;
    double t_start = t_data.front();
    double t_end = t_data.back();
    
    cv::Point prev_point;
    for (int i = 0; i < 200; ++i) { //形成平滑曲线
        double t = t_start + (t_end - t_start) * i / 199.0;
        double pred_x, pred_y;
        model(fit_params, t, &pred_x, &pred_y);

        cv::Point current_point(cv::saturate_cast<int>(pred_x), 
                                cv::saturate_cast<int>(img_height - pred_y));
        
        if (i > 0) {
            cv::line(result_image, prev_point, current_point, cv::Scalar(0, 0, 255), 2);
        }
        prev_point = current_point;
    }
    
    //图例
    cv::putText(result_image, "Observed Points", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
    cv::putText(result_image, "Fitted Trajectory", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

    //显示图像
    const std::string output_filename = "result_fit.png";
    cv::imshow("Trajectory Fit Result", result_image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    //提取数据
    std::vector<double> t_data, x_data, y_data;
    cv::Mat background_frame; //存储背景图像
    if (!extract_trajectory_from_video(VIDEO_FILE_PATH, t_data, x_data, y_data, background_frame)) {
        return -1;
    }

    //Ceres优化
    double initial_x0 = x_data[0];
    double initial_y0 = y_data[0];
    double initial_v0x = (x_data.size() > 1) ? (x_data[1] - x_data[0]) / (t_data[1] - t_data[0]) : 50.0;
    double initial_v0y = (y_data.size() > 1) ? (y_data[1] - y_data[0]) / (t_data[1] - t_data[0]) : 50.0;
    double params[6] = {initial_x0, initial_y0, initial_v0x, initial_v0y, 500.0, 0.1};
    
    ceres::Problem problem;
    for (size_t i = 0; i < t_data.size(); ++i) {
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<BallisticCostFunctor, 2, 6>(
            new BallisticCostFunctor(t_data[i], x_data[i], y_data[i])), nullptr, params);
    }
    problem.SetParameterLowerBound(params, 4, 100.0);
    problem.SetParameterUpperBound(params, 4, 1000.0);
    problem.SetParameterLowerBound(params, 5, 0.01);
    problem.SetParameterUpperBound(params, 5, 1.0);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    std::cout << "\n";
    std::cout << "Final Estimated Parameters:\n";
    printf("Initial Position (x₀, y₀)   = (%.2f, %.2f) px\n", params[0], params[1]);
    printf("Initial Velocity (v₀x, v₀y) = (%.2f, %.2f) px/s\n", params[2], params[3]);
    printf("Gravity (g)                 = %.2f px/s²\n", params[4]);
    printf("Air Resistance (k)          = %.4f 1/s\n", params[5]);

    //可视化结果
    if (!background_frame.empty()) {
        visualize_and_save_results(background_frame, t_data, x_data, y_data, params);
    }

    return 0;
}