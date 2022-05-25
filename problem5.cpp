#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <opencv2/core/eigen.hpp>
#include "sophus/se3.hpp"
#include "rotation.h"
#include "parametersse3.hpp"

using namespace std;
using namespace cv;

inline cv::Scalar get_color(float depth)
{
  float up_th = 50, low_th = 10, th_range = up_th - low_th;
  if (depth > up_th)
    depth = up_th;
  if (depth < low_th)
    depth = low_th;
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches,
                          int nfeatures);
void ransacTest(const std::vector<cv::DMatch> &matches,
                std::vector<cv::KeyPoint> &keypoints1,
                std::vector<cv::KeyPoint> &keypoints2,
                std::vector<cv::DMatch> &outMatches,
                int ransactheshold);
cv::Mat compute_fundamental_matrix(const std::vector<cv::DMatch> &matches,
                                   std::vector<cv::KeyPoint> &keypoints1,
                                   std::vector<cv::KeyPoint> &keypoints2,
                                   const Eigen::Matrix3d &K);
Point2d pixel2cam(const Point2d &p, const Mat &K);
Point2d cam2pixel(const Point2d &p, const Mat &K);
void triangulation(
    const vector<KeyPoint> &keypoint_1,
    const vector<KeyPoint> &keypoint_2,
    const std::vector<DMatch> &matches,
    const Mat &R, const Mat &t,
    vector<Point3d> &points,
    const cv::Mat &K);
void triangulation_opencv(
    const vector<KeyPoint> &keypoint_1,
    const vector<KeyPoint> &keypoint_2,
    const std::vector<DMatch> &matches,
    const Mat &R, const Mat &t,
    vector<Point3d> &points_3d,
    const cv::Mat &K);
void Triangulate(
    const cv::KeyPoint &kp1, //特征点, in reference frame
    const cv::KeyPoint &kp2, //特征点, in current frame
    const cv::Mat &P1,       //投影矩阵P1
    const cv::Mat &P2,       //投影矩阵P2
    cv::Mat &x3D);

int main(int agrc, char **argv)
{
  string file_name = "/home/docker_file/Nreal_training/two_image_pose_estimation/1403637188088318976.png";
  string file_name2 = "/home/docker_file/Nreal_training/two_image_pose_estimation/1403637189138319104.png";
  string camera_yaml_filename = "/home/docker_file/Nreal_training/two_image_pose_estimation/sensor.yaml";

  Mat picture_1 = imread(file_name, IMREAD_UNCHANGED);
  Mat picture_2 = imread(file_name2, IMREAD_UNCHANGED);

  Mat picture_1_undistort, picture_2_undistort, K_cv;
  //
  // 457.296
  double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
  Eigen::Matrix3d K;
  K << fx, 0, cx,
      0, fy, cy,
      0, 0, 1;
  Mat R_cv, Distcoef(4, 1, CV_32F);

  Distcoef.at<float>(0) = -0.28340811;
  Distcoef.at<float>(1) = 0.07395907;
  Distcoef.at<float>(2) = 0.00019359;
  Distcoef.at<float>(3) = 1.76187114e-05;

  eigen2cv(K, K_cv);
  undistort(picture_1, picture_1_undistort, K_cv, Distcoef);
  undistort(picture_2, picture_2_undistort, K_cv, Distcoef);
  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches, matches_after_ransac;
  int nfeatures = 1000;
  int ransac_reprojection = 3;
  find_feature_matches(picture_1_undistort, picture_2_undistort, keypoints_1, keypoints_2, matches, nfeatures);
  ransacTest(matches, keypoints_1, keypoints_2, matches_after_ransac, ransac_reprojection);

  Mat img_match, img_ransacmatch;
  drawMatches(picture_1_undistort, keypoints_1, picture_2_undistort, keypoints_2, matches, img_match);
  drawMatches(picture_1_undistort, keypoints_1, picture_2_undistort, keypoints_2, matches_after_ransac, img_ransacmatch);

  // imshow("all matches", img_match);
  // imshow("ransac matches", img_ransacmatch);
  // waitKey(0);

  cv::Mat F_estimate = compute_fundamental_matrix(matches_after_ransac, keypoints_1, keypoints_2, K);
  cout << "our fundamental matrix\n"
       << F_estimate << endl;

  vector<Point2f> points_1, points_2;
  for (int i = 0; i < matches_after_ransac.size(); i++)
  {
    points_1.push_back(keypoints_1[matches_after_ransac[i].queryIdx].pt);
    points_2.push_back(keypoints_2[matches_after_ransac[i].trainIdx].pt);
  }

  std::vector<uchar> inliers(points_1.size(), 0);
  cv::Mat F_opencv = findFundamentalMat(points_1, points_2, inliers, // 匹配状态(inlier 或 outlier)
                                        cv::FM_RANSAC,
                                        ransac_reprojection,
                                        0.99);
  vector<Point2f> points_1_new, points_2_new;
  // correctMatches(F_opencv, points_1, points_2, points_1_new, points_2_new);

  cv::Mat E_opencv = K_cv.t() * F_opencv * K_cv;
  cout << "F_opencv\n"
       << F_opencv << endl;
  cv::Mat R_cv_estimated, t_cv_estimated;
  cv::Mat E_opencv2 = findEssentialMat(points_1, points_2, K_cv);
  recoverPose(E_opencv, points_1, points_2, K_cv, R_cv_estimated, t_cv_estimated);
  cout << "R_CV_estimated" << R_cv_estimated << endl;
  vector<cv::Point3d> pts_3d;
  triangulation_opencv(keypoints_1, keypoints_2, matches_after_ransac, R_cv_estimated, t_cv_estimated, pts_3d, K_cv);
  
  //*优化之前的重投影误差计算
  double error_before_BA = 0;
  vector<double> verror_before_BA(pts_3d.size(), 0);
  for (int i = 0; i < pts_3d.size(); i++)
  {

    Mat P3D = (Mat_<double>(3, 1) << pts_3d[i].x, pts_3d[i].y, pts_3d[i].z);
    // cout<<"t_cv_estimated\n"<<t_cv_estimated<<endl;
    Mat pts_3d_img2 = R_cv_estimated * P3D + t_cv_estimated;

    double xp = pts_3d_img2.at<double>(0) / pts_3d_img2.at<double>(2);
    double yp = pts_3d_img2.at<double>(1) / pts_3d_img2.at<double>(2);

    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
    double u = xp * fx + cx;
    double v = yp * fy + cy;
    double dx = points_2[i].x - u;
    double dy = points_2[i].y - v;

    Eigen::Vector2d error_point(dx, dy);
    // cout << "第" << i << "个点的重投影误差" << error_point.norm() << endl;
    error_before_BA += error_point.norm();
    verror_before_BA.push_back(error_point.norm());
  }

  float mean_error_before_BA = error_before_BA / pts_3d.size() * 1.0;
  cout << " 优化前平均投影误差:" << mean_error_before_BA << endl;
  double variance_befor_BA = 0;
  for (auto i : verror_before_BA)
    variance_befor_BA += (i - mean_error_before_BA) * (i - mean_error_before_BA);
  variance_befor_BA /= verror_before_BA.size();
  cout << "variance_befor_BA:" << variance_befor_BA << endl;
  cout << "Standard Deviation before BA:" << sqrt(variance_befor_BA) << endl;

  vector<double> r_vector, t_vector(3);
  double cere_rot[3], cere_tranf[3];
  cv::Rodrigues(R_cv_estimated, r_vector);

  t_vector[0] = t_cv_estimated.at<double>(0, 0);
  t_vector[1] = t_cv_estimated.at<double>(1, 0);
  t_vector[2] = t_cv_estimated.at<double>(2, 0);
  //构造旋转向量+平移向量作为待优化变量
  double camera_6[6];
  camera_6[0] = r_vector[0];
  camera_6[1] = r_vector[1];
  camera_6[2] = r_vector[2];
  camera_6[3] = t_vector[0];
  camera_6[4] = t_vector[1];
  camera_6[5] = t_vector[2];

  double camera_6_2_auto[6];
  camera_6_2_auto[0] = r_vector[0];
  camera_6_2_auto[1] = r_vector[1];
  camera_6_2_auto[2] = r_vector[2];
  camera_6_2_auto[3] = t_vector[0];
  camera_6_2_auto[4] = t_vector[1];
  camera_6_2_auto[5] = t_vector[2];

  //构造四元数+平移向量作为待优化变量
  double camera_7[7];
  Eigen::Map<Eigen::Quaterniond> q_tmp(camera_7);
  Eigen::Map<Eigen::Vector3d> t_tmp(camera_7 + 4);
  Eigen::Vector3d r_eigen;
  r_eigen << r_vector[0], r_vector[1], r_vector[2];
  t_tmp << t_vector[1], t_vector[1], t_vector[2];
  q_tmp = toQuaterniond(r_eigen);

  //构造位姿李代数作为优化变量
  Eigen::Matrix3d R_eigen;
  Eigen::Vector3d t_eigen;
  cv2eigen(R_cv_estimated,R_eigen);
  cv2eigen(t_cv_estimated,t_eigen);
  Sophus::SE3d T(R_eigen,t_eigen);
  Eigen::Matrix<double,6,1> se3(T.log());

  //地图点作为待优化变量
  vector<Point2f> &pts_2d = points_2;
  vector<Eigen::Vector3d> pts_3d_eigen, pts_3d_eigen_for_quaternion, pts_3d_eigen_auto,pts_3d_for_se3;
  vector<Eigen::Vector2d> pts_2d_eigen;

  for (int i = 0; i < pts_2d.size(); i++)
  {
    pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_3d_eigen_for_quaternion.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_3d_eigen_auto.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_3d_for_se3.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
  }

  ceres::Problem problem_auto;

  for (int i = 0; i < pts_2d_eigen.size(); ++i)
  {

    ceres::CostFunction *costFunc = ReprojectionError::Create(pts_2d_eigen[i].x(), pts_2d_eigen[i].y());
    problem_auto.AddResidualBlock(costFunc, NULL, camera_6_2_auto, pts_3d_eigen_auto[i].data());
  }
  auto t1_auto_2 = chrono::steady_clock::now();

  ceres::Solver::Options options_auto;
  ceres::Solver::Summary summary_auto;

  options_auto.linear_solver_type = ceres::SPARSE_SCHUR;
  options_auto.minimizer_progress_to_stdout = true;
  options_auto.max_num_iterations = 50;
  options_auto.check_gradients = false;

  ceres::Solve(options_auto, &problem_auto, &summary_auto);
  cout << summary_auto.FullReport() << endl;
  auto t2_auto_2 = chrono::steady_clock::now();
  auto time_auto_used_2 = chrono::duration_cast<chrono::duration<double>>(t2_auto_2 - t1_auto_2);
  cout << "ceres自动求导(旋转向量)耗时: " << time_auto_used_2.count() << " seconds." << endl;

  //记录重投影误差
  vector<double> verror_auto_pointpose_BA;
  double error_auto_pointpose_BA = 0;
  double variance_auto_pointpose_BA = 0, mean_error_auto_pointpose_BA = 0;

  for (int i = 0; i < pts_3d_eigen_auto.size(); i++)
  {
    Eigen::Map<Eigen::Vector3d> r_vector_after_optimized(camera_6_2_auto);
    Eigen::Map<Eigen::Vector3d> t_vector_after_optimized(camera_6_2_auto + 3);

    auto q_after_optimized = toQuaterniond(r_vector_after_optimized);

    Sophus::SE3d T(q_after_optimized, t_vector_after_optimized);
    Eigen::Vector3d Pc = T * pts_3d_eigen_auto[i];
    Eigen::Vector2d error = pts_2d_eigen[i] - (K * Pc).hnormalized();

    cout << "第" << i << "个点的重投影误差" << error.norm() << endl;
    error_auto_pointpose_BA += error.norm();
    verror_auto_pointpose_BA.push_back(error.norm());
  }

  mean_error_auto_pointpose_BA = error_auto_pointpose_BA / pts_3d_eigen_auto.size();
  cout << "自动求导平均投影误差" << mean_error_auto_pointpose_BA << endl;

  for (auto i : verror_auto_pointpose_BA)
    variance_auto_pointpose_BA += (i - mean_error_auto_pointpose_BA) * (i - mean_error_auto_pointpose_BA);

  variance_auto_pointpose_BA /= verror_auto_pointpose_BA.size();

  cout << "variance_auto_pointpose_BA:" << variance_auto_pointpose_BA << endl;
  cout << "Standard Deviation auto pointpose BA:" << sqrt(variance_auto_pointpose_BA) << endl;

  //*手动求导
  //旋转向量
  ceres::Problem problem_;
  // vector<const ceres::LocalParameterization *> localparameters;
  // // ceres::LocalParameterization *local_param = new PoseSE3Parameterization<6>();
  // // localparameters.push_back(local_param);
  // // localparameters.push_back(nullptr);
  // ceres::NumericDiffOptions numeric_diff_options;

  problem_.AddParameterBlock(camera_6, 6, new PoseSE3Parameterization<6>());

  for (int i = 0; i < pts_2d_eigen.size(); ++i)
  {
    ceres::CostFunction *costFunc = new ReprojectionErrorSE3XYZ<6>(fx, cx, cy, pts_2d_eigen[i].x(), pts_2d_eigen[i].y());

    problem_.AddParameterBlock(pts_3d_eigen[i].data(), 3);
    problem_.AddResidualBlock(costFunc, NULL, camera_6, pts_3d_eigen[i].data());
    // ceres::GradientChecker gradient_checker(costFunc, &localparameters, numeric_diff_options);
    // std::vector<double *> parameter_blocks;
    // parameter_blocks.push_back(camera);
    // parameter_blocks.push_back(pts_3d_eigen[i].data());
    // ceres::GradientChecker::ProbeResults results;
    //  if (!gradient_checker.Probe(parameter_blocks.data(), 1e-4, &results))
    //  {
    //    LOG(ERROR) << "An error has occurred:\n"
    //               << results.error_log;
    //  }
  }

  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;
  options_.linear_solver_type = ceres::SPARSE_SCHUR;
  options_.minimizer_progress_to_stdout = true;
  options_.max_num_iterations = 50;

  //梯度检查
  // options_.check_gradients = true;

  auto t1_manual = chrono::steady_clock::now();
  ceres::Solve(options_, &problem_, &summary_);
  // std::cout << summary_.BriefReport() << "\n";
  std::cout << summary_.BriefReport() << "\n";
  auto t2_manual = chrono::steady_clock::now();
  auto time_used = chrono::duration_cast<chrono::duration<double>>(t2_manual - t1_manual);
  cout << "ceres手动求导耗时: " << time_used.count() << " seconds." << endl;

  //*重投影误差计算

  vector<double> verror_manunal_pointpose_BA;
  double error_manunal_pointpose_BA = 0;
  double variance_manunal_pointpose_BA = 0, mean_error_manunal_pointpose_BA = 0;

  for (int i = 0; i < pts_3d_eigen.size(); i++)
  {
    Eigen::Map<Eigen::Vector3d> r_vector_after_optimized(camera_6);
    Eigen::Map<Eigen::Vector3d> t_vector_after_optimized(camera_6 + 3);

    auto q_after_optimized = toQuaterniond(r_vector_after_optimized);

    Sophus::SE3d T(q_after_optimized, t_vector_after_optimized);
    Eigen::Vector3d Pc = T * pts_3d_eigen[i];
    Eigen::Vector2d error = pts_2d_eigen[i] - (K * Pc).hnormalized();

    // cout << "第" << i << "个点的重投影误差" << error.norm() << endl;
    error_manunal_pointpose_BA += error.norm();
    verror_manunal_pointpose_BA.push_back(error.norm());
  }

  mean_error_manunal_pointpose_BA = error_manunal_pointpose_BA / pts_3d_eigen.size();
  cout << "平均投影误差" << mean_error_manunal_pointpose_BA << endl;

  for (auto i : verror_manunal_pointpose_BA)
    variance_manunal_pointpose_BA += (i - mean_error_manunal_pointpose_BA) * (i - mean_error_manunal_pointpose_BA);

  variance_manunal_pointpose_BA /= verror_manunal_pointpose_BA.size();

  cout << "variance_manunal_pointpose_BA:" << variance_manunal_pointpose_BA << endl;
  cout << "Standard Deviation manunal pointpose BA:" << sqrt(variance_manunal_pointpose_BA) << endl;

  ceres::Problem problem_2;
  problem_2.AddParameterBlock(camera_7, 7, new PoseSE3Parameterization<7>());
  for (int i = 0; i < pts_2d_eigen.size(); ++i)
  {
    ceres::CostFunction *costFunc = new ReprojectionErrorSE3XYZ<7>(fx, cx, cy, pts_2d_eigen[i].x(), pts_2d_eigen[i].y());
    problem_2.AddParameterBlock(pts_3d_eigen_for_quaternion[i].data(), 3);
    problem_2.AddResidualBlock(costFunc, NULL, camera_7, pts_3d_eigen_for_quaternion[i].data());
  }

  ceres::Solver::Options options_2;
  ceres::Solver::Summary summary_2;
  options_2.linear_solver_type = ceres::SPARSE_SCHUR;
  options_2.minimizer_progress_to_stdout = true;
  options_2.max_num_iterations = 50;
  options_2.check_gradients = false;

  auto t1_manual_2 = chrono::steady_clock::now();
  ceres::Solve(options_2, &problem_2, &summary_2);
  // std::cout << summary_.BriefReport() << "\n";
  std::cout << summary_2.FullReport() << "\n";
  auto t2_manual_2 = chrono::steady_clock::now();
  auto time_used_2 = chrono::duration_cast<chrono::duration<double>>(t2_manual_2 - t1_manual_2);
  cout << "ceres手动求导(四元数)耗时: " << time_used_2.count() << " seconds." << endl;

  vector<double> verror_manunal_pointpose_BA_2;
  double error_manunal_pointpose_BA_2 = 0;
  double variance_manunal_pointpose_BA_2 = 0, mean_error_manunal_pointpose_BA_2 = 0;

  ofstream reprojection_x("/home/docker_file/Nreal_training/x.txt");
  ofstream reprojection_y("/home/docker_file/Nreal_training/y.txt");

  for (int i = 0; i < pts_3d_eigen.size(); i++)
  {
    Eigen::Map<Eigen::Quaterniond> q_after_optimized(camera_7);
    Eigen::Map<Eigen::Vector3d> t_vector_after_optimized(camera_7 + 4);

    Sophus::SE3d T(q_after_optimized, t_vector_after_optimized);
    Eigen::Vector3d Pc = T * pts_3d_eigen_for_quaternion[i];
    Eigen::Vector2d error = pts_2d_eigen[i] - (K * Pc).hnormalized();

    reprojection_x << error.x() << endl;
    reprojection_y << error.y() << endl;

    // cout << "第" << i << "个点的重投影误差" << error.norm() << endl;
    error_manunal_pointpose_BA_2 += error.norm();
    verror_manunal_pointpose_BA_2.push_back(error.norm());
  }

  mean_error_manunal_pointpose_BA_2 = error_manunal_pointpose_BA_2 / pts_3d_eigen_for_quaternion.size();
  cout << "手动求导平均投影误差" << mean_error_manunal_pointpose_BA_2 << endl;

  for (auto i : verror_manunal_pointpose_BA_2)
    variance_manunal_pointpose_BA_2 += (i - mean_error_manunal_pointpose_BA_2) * (i - mean_error_manunal_pointpose_BA_2);

  variance_manunal_pointpose_BA_2 /= verror_manunal_pointpose_BA_2.size();
  cout << "variance_manunal_pointpose_BA:" << variance_manunal_pointpose_BA_2 << endl;
  cout << "Standard Deviation manunal pointpose BA:" << sqrt(variance_manunal_pointpose_BA_2) << endl;



  //*关于位姿李代数的导数
  ceres::Problem problem_3;
  problem_3.AddParameterBlock(se3.data(),6,new Pose_se3Parameterization());
  for (int i = 0; i < pts_2d_eigen.size(); ++i)
  {
    ceres::CostFunction *costFunc = new ReprojectionErrorse3(fx, cx, cy, pts_2d_eigen[i].x(), pts_2d_eigen[i].y());
    problem_3.AddParameterBlock(pts_3d_for_se3[i].data(), 3);
    problem_3.AddResidualBlock(costFunc, NULL, se3.data(), pts_3d_for_se3[i].data());
  }
  ceres::Solver::Options options_3;
  ceres::Solver::Summary summary_3;
  options_3.linear_solver_type = ceres::SPARSE_SCHUR;
  options_3.minimizer_progress_to_stdout = true;
  options_3.max_num_iterations = 50;
  options_3.check_gradients = true;
  ceres::Solve(options_3, &problem_3, &summary_3);
  cout<<summary_3.FullReport()<<endl;

  //* 重投影图片可视化
  // Mat img1_plot = picture_1_undistort.clone();
  // Mat img2_plot = picture_2_undistort.clone();
  // vector<KeyPoint> keypoint_observerd(matches_after_ransac.size(), KeyPoint());
  // vector<KeyPoint> keypoint_projectd(matches_after_ransac.size(), KeyPoint());
  // for (int i = 0; i < matches_after_ransac.size(); i++)
  // {
  //   float depth1 = pts_3d_eigen[i].z();
  //   double xp = pts_3d_eigen[i].x() / depth1;
  //   double yp = pts_3d_eigen[i].y() / depth1;

  //   Point2f p_projection = cam2pixel(Point2f(xp, yp), K_cv);
  //   keypoint_observerd[i].pt = keypoints_1[matches_after_ransac[i].queryIdx].pt;
  //   keypoint_projectd[i].pt.x = p_projection.x;
  //   keypoint_projectd[i].pt.y = p_projection.y;
  //   // cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, cv::Scalar(255,0,0), 2);
  //   // cv::circle(img1_plot, p_projection, 2, cv::Scalar(255,255,255), 2);
  // }
  // drawKeypoints(img1_plot, keypoint_observerd, img1_plot, cv::Scalar(255, 0, 0));
  // drawKeypoints(img1_plot, keypoint_projectd, img1_plot, cv::Scalar(0, 0, 255));

  // cv::imshow("投影点和提取点差异 1", img1_plot);
  // cv::waitKey(0);
  //   return 0;

}

cv::Mat compute_fundamental_matrix(const std::vector<cv::DMatch> &matches,
                                   std::vector<cv::KeyPoint> &keypoints1,
                                   std::vector<cv::KeyPoint> &keypoints2,
                                   const Eigen::Matrix3d &K)
{
  int A_rows = matches.size();
  Eigen::MatrixXd A(A_rows, 9);
  for (int i = 0; i < A_rows; i++)
  {
    int u1, u2, v1, v2;
    u1 = keypoints1[matches[i].queryIdx].pt.x;
    v1 = keypoints1[matches[i].queryIdx].pt.y;
    u2 = keypoints2[matches[i].trainIdx].pt.x;
    v2 = keypoints2[matches[i].trainIdx].pt.y;
    A(i, 0) = u2 * u1;
    A(i, 1) = u2 * v1;
    A(i, 2) = u2;
    A(i, 3) = v2 * u1;
    A(i, 4) = v2 * v1;
    A(i, 5) = v2;
    A(i, 6) = u1;
    A(i, 7) = v1;
    A(i, 8) = 1;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

  Eigen::VectorXd F_vector = svd.matrixV().col(8);
  Eigen::Matrix3d F;
  F << F_vector(0), F_vector(1), F_vector(2),
      F_vector(3), F_vector(4), F_vector(5),
      F_vector(6), F_vector(7), F_vector(8);
  cout << "F矩阵:\n"
       << F << endl;

  Eigen::JacobiSVD<Eigen::MatrixXd> svd2(F, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::MatrixXd U = svd2.matrixU();
  Eigen::MatrixXd V = svd2.matrixV();
  Eigen::Vector3d w = svd2.singularValues();
  w(2) = 0;
  Eigen::Matrix3d w_matrix(w.asDiagonal());
  //新F
  Mat F_cv;
  F = U * w_matrix * V.transpose();
  eigen2cv(F, F_cv);
  return F_cv;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches, int nfeatures)
{
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create(nfeatures);
  Ptr<DescriptorExtractor> descriptor = ORB::create(nfeatures);
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  vector<DMatch> match;
  matcher->match(descriptors_1, descriptors_2, match);

  double min_dist = 10000, max_dist = 0;

  for (int i = 0; i < descriptors_1.rows; i++)
  {
    double dist = match[i].distance;
    if (dist < min_dist)
      min_dist = dist;
    if (dist > max_dist)
      max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  for (int i = 0; i < descriptors_1.rows; i++)
  {
    if (match[i].distance <= max(2 * min_dist, 30.0))
    {
      matches.push_back(match[i]);
    }
  }
}
void ransacTest(const std::vector<cv::DMatch> &matches,
                std::vector<cv::KeyPoint> &keypoints1,
                std::vector<cv::KeyPoint> &keypoints2,
                std::vector<cv::DMatch> &outMatches, int ransactheshold)
{

  // 将关键点转换为 Point2f 类型
  std::vector<cv::Point2f> points1, points2;

  for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
       it != matches.end(); ++it)
  {

    // 获取左侧关键点的位置
    points1.push_back(keypoints1[it->queryIdx].pt);
    // 获取右侧关键点的位置
    points2.push_back(keypoints2[it->trainIdx].pt);
  }
  int distance = ransactheshold;
  // 用 RANSAC 计算 F 矩阵
  std::vector<uchar> inliers(points1.size(), 0);
  cv::Mat fundamental = cv::findFundamentalMat(
      points1, points2, // 匹配像素点
      inliers,          // 匹配状态(inlier 或 outlier)
      cv::FM_RANSAC,    // RANSAC 算法
      distance,         // 到对极线的距离
      0.99);            // 置信度

  std::vector<uchar>::const_iterator itIn = inliers.begin();
  std::vector<cv::DMatch>::const_iterator itM = matches.begin();
  for (; itIn != inliers.end(); ++itIn, ++itM)
  {
    if (*itIn)
    {
      outMatches.push_back(*itM);
    }
  }
}
Point2d pixel2cam(const Point2d &p, const Mat &K)
{
  return Point2d(
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}
Point2d cam2pixel(const Point2d &p, const Mat &K)
{
  return Point2d(
      (p.x * K.at<double>(0, 0) + K.at<double>(0, 2)),
      (p.y * K.at<double>(1, 1) + K.at<double>(1, 2)));
}

void triangulation_opencv(
    const vector<KeyPoint> &keypoint_1,
    const vector<KeyPoint> &keypoint_2,
    const std::vector<DMatch> &matches,
    const Mat &R, const Mat &t,
    vector<Point3d> &points_3d,
    const cv::Mat &K)
{
  Mat T1 = (Mat_<float>(3, 4) << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0);
  Mat T2 = (Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

  vector<Point2f> pts_1, pts_2;
  for (DMatch m : matches)
  {
    // 将像素坐标转换至相机坐标
    pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
    pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
  }

  Mat pts_4d;
  cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

  // 转换成非齐次坐标
  for (int i = 0; i < pts_4d.cols; i++)
  {
    Mat x = pts_4d.col(i);
    x /= x.at<float>(3, 0); // 归一化
    Point3d p(
        x.at<float>(0, 0),
        x.at<float>(1, 0),
        x.at<float>(2, 0));
    points_3d.push_back(p);
  }
}
void triangulation(
    const vector<KeyPoint> &keypoint_1,
    const vector<KeyPoint> &keypoint_2,
    const std::vector<DMatch> &matches,
    const Mat &R, const Mat &t,
    vector<Point3d> &points,
    const cv::Mat &K)
{
  Eigen::Matrix3d R_eign;
  Eigen::Vector3d t_eigen;
  cv2eigen(R, R_eign);
  cv2eigen(t, t_eigen);
  Mat T1 = (Mat_<float>(3, 4) << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0);
  Mat T2 = (Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
  Eigen::MatrixXd A(matches.size(), 2);
  vector<Point2f> pts_1, pts_2;
  // todo 这里有一个bug,特征点不会都是一个深度
  //只是最小二乘意义下的解，而不是超定方程的解
  for (int i = 0; i < matches.size() - 3; i += 3)
  {
    // 将像素坐标转换至归一化平面坐标

    Point2f x1(pixel2cam(keypoint_1[matches[i].queryIdx].pt, K));
    Point2f x2(pixel2cam(keypoint_2[matches[i].trainIdx].pt, K));
    Eigen::Vector3d x1_eigen(x1.x, x1.y, 1);
    pts_1.push_back(x1);
    pts_2.push_back(x2);
    Eigen::Matrix3d x2_hat;
    x2_hat << 0, -1, x2.y,
        1, 0, -x2.x,
        -x2.y, x2.x, 0;
    Eigen::MatrixXd A_sub(3, 2);
    A_sub.col(0) = x2_hat * R_eign * x1_eigen;
    A_sub.col(1) = x2_hat * t_eigen;
    A.row(i) = A_sub.row(0);
    A.row(i + 1) = A_sub.row(1);
    A.row(i + 2) = A_sub.row(2);
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  int V_last_cols = svd.matrixV().cols();
  Eigen::Vector2d s = svd.matrixV().col(V_last_cols - 1);
  cout << "\ns\n"
       << s << endl;
}
void triangulation2(
    const vector<KeyPoint> &keypoint_1,
    const vector<KeyPoint> &keypoint_2,
    const std::vector<DMatch> &matches,
    const Mat &R, const Mat &t,
    vector<Point3d> &points,
    const cv::Mat &K)
{
  int nbgood = 0;
  cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));

  Mat P2 = (Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
  P2 = K * P2;
  for (size_t i = 0, iend = matches.size(); i < iend; i++)
  {
    const cv::KeyPoint &kp1 = keypoint_1[matches[i].queryIdx];
    const cv::KeyPoint &kp2 = keypoint_2[matches[i].trainIdx];
    cv::Mat p3dC1;
    Triangulate(kp1, kp2, //特征点
                P1, P2,   //投影矩阵
                p3dC1);
    if (p3dC1.at<float>(2) <= 0)
      continue;

    // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
    // 讲空间点p3dC1变换到第2个相机坐标系下变为p3dC2
    cv::Mat p3dC2 = R * p3dC1 + t;
    //判断过程和上面的相同
    if (p3dC2.at<float>(2) <= 0)
      continue;

    float im1x, im1y;
    //这个使能空间点的z坐标的倒数
    float invZ1 = 1.0 / p3dC1.at<float>(2);

    //投影到参考帧图像上。因为参考帧下的相机坐标系和世界坐标系重合，因此这里就直接进行投影就可以了
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
    im1x = fx * p3dC1.at<float>(0) * invZ1 + cx;
    im1y = fy * p3dC1.at<float>(1) * invZ1 + cy;
    //参考帧上的重投影误差，这个的确就是按照定义来的
    float squareError1 = (im1x - kp1.pt.x) * (im1x - kp1.pt.x) + (im1y - kp1.pt.y) * (im1y - kp1.pt.y);
    if (squareError1 > 4)
      continue;
    float im2x, im2y;
    // 注意这里的p3dC2已经是第二个相机坐标系下的三维点了
    float invZ2 = 1.0 / p3dC2.at<float>(2);
    im2x = fx * p3dC2.at<float>(0) * invZ2 + cx;
    im2y = fy * p3dC2.at<float>(1) * invZ2 + cy;

    // 计算重投影误差
    float squareError2 = (im2x - kp2.pt.x) * (im2x - kp2.pt.x) + (im2y - kp2.pt.y) * (im2y - kp2.pt.y);
    // 重投影误差太大，跳过淘汰
    if (squareError2 > 4)
      continue;
    points[matches[i].queryIdx] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
    nbgood++;
  }
}

void Triangulate(
    const cv::KeyPoint &kp1, //特征点, in reference frame
    const cv::KeyPoint &kp2, //特征点, in current frame
    const cv::Mat &P1,       //投影矩阵P1
    const cv::Mat &P2,       //投影矩阵P2
    cv::Mat &x3D)            //三维点
{
  // 原理
  // Trianularization: 已知匹配特征点对{x x'} 和 各自相机矩阵{P P'}, 估计三维点 X
  // x' = P'X  x = PX
  // 它们都属于 x = aPX模型
  //                         |X|
  // |x|     |p1 p2  p3  p4 ||Y|     |x|    |--p0--||.|
  // |y| = a |p5 p6  p7  p8 ||Z| ===>|y| = a|--p1--||X|
  // |z|     |p9 p10 p11 p12||1|     |z|    |--p2--||.|
  // 采用DLT的方法：x叉乘PX = 0
  // |yp2 -  p1|     |0|
  // |p0 -  xp2| X = |0|
  // |xp1 - yp0|     |0|
  // 两个点:
  // |yp2   -  p1  |     |0|
  // |p0    -  xp2 | X = |0| ===> AX = 0
  // |y'p2' -  p1' |     |0|
  // |p0'   - x'p2'|     |0|
  // 变成程序中的形式：
  // |xp2  - p0 |     |0|
  // |yp2  - p1 | X = |0| ===> AX = 0
  // |x'p2'- p0'|     |0|
  // |y'p2'- p1'|     |0|
  // 然后就组成了一个四元一次正定方程组，SVD求解，右奇异矩阵的最后一行就是最终的解.

  //这个就是上面注释中的矩阵A
  cv::Mat A(4, 4, CV_32F);

  //构造参数矩阵A
  A.row(0) = kp1.pt.x * P1.row(2) - P1.row(0);
  A.row(1) = kp1.pt.y * P1.row(2) - P1.row(1);
  A.row(2) = kp2.pt.x * P2.row(2) - P2.row(0);
  A.row(3) = kp2.pt.y * P2.row(2) - P2.row(1);

  //奇异值分解的结果
  cv::Mat u, w, vt;
  //对系数矩阵A进行奇异值分解
  cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  //根据前面的结论，奇异值分解右矩阵的最后一行其实就是解，原理类似于前面的求最小二乘解，四个未知数四个方程正好正定
  //别忘了我们更习惯用列向量来表示一个点的空间坐标
  x3D = vt.row(3).t();
  //为了符合其次坐标的形式，使最后一维为1
  x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
}
