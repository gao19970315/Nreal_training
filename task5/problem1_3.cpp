#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>


using namespace std;

int main(int agrc, char **argv)
{
    string file_name = "/home/docker_file/Nreal_training/two_image_pose_estimation/1403637188088318976.png";
    string camera_yaml_filename = "/home/docker_file/Nreal_training/two_image_pose_estimation/sensor.yaml";
    string output_filename = "/home/docker_file/Nreal_training/two_image_pose_estimation/output.png";
    cv::Mat picture_1 = cv::imread(file_name, cv::IMREAD_UNCHANGED);
    cv::Mat picture_1_undistort,picture_1_undistort_2,K_cv;
    cv::Mat picture_1_test=picture_1.clone();

    int picture_1_width = picture_1.cols;
    int picture_1_height = picture_1.rows;

    for (int row = 0; row < picture_1_height; row++)
    {
        uchar *p_picture_1_pointer = picture_1_test.ptr(row);
        for (int col = 0; col < picture_1_width; col++)
        {
            *p_picture_1_pointer = 255 - *p_picture_1_pointer;
            p_picture_1_pointer++;
        }
    }
    cv::imshow("picture1",picture_1_test);
    cv::imwrite(output_filename,picture_1_test);
    cv::waitKey(0);
    

    Eigen::Matrix3d K;
    
    Eigen::Matrix4d tmp;
    K << 458.654, 0, 367.215,
        0, 457.296, 248.375,
        0, 0, 1;
    tmp << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
        0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
        0.0, 0.0, 0.0, 1.0;
    Eigen::Isometry3d T(tmp);
    cv::Mat R_cv,Distcoef(4, 1, CV_32F);
    cv::eigen2cv(T.rotation(),R_cv);

    Distcoef.at<float>(0) = -0.28340811;
    Distcoef.at<float>(1) = 0.07395907;
    Distcoef.at<float>(2) = 0.00019359;
    Distcoef.at<float>(3) = 1.76187114e-05;
    cv::eigen2cv(K,K_cv);
    cv::undistort(picture_1,picture_1_undistort, K_cv,Distcoef);
    cv::imshow("undistort:undistort",picture_1_undistort);
    cv::imshow("distort",picture_1);

    cv::Mat map_x,map_y,k_cv_new;
    cv::Size picture_1_size(picture_1.size());
    int alpha=0;
    k_cv_new=cv::getOptimalNewCameraMatrix(K_cv,Distcoef,picture_1_size,alpha,picture_1_size);
    cout<<k_cv_new;
    cv::initUndistortRectifyMap(K_cv,Distcoef,cv::Mat(),k_cv_new,picture_1_size,CV_32FC1,map_x,map_y);
    cv::remap(picture_1,picture_1_undistort_2,map_x,map_y,cv::INTER_LINEAR);
    cv::imshow("undistort2:remap",picture_1_undistort_2);
    //cv::undistortPoints
    cv::waitKey(0);
}
