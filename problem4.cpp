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

using namespace std;
using namespace cv;

class SE3Parameterization : public ceres::LocalParameterization
{
public:
    SE3Parameterization() {}
    virtual ~SE3Parameterization() {}

    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const
    {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(x);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_lie(delta);

        Sophus::SE3d T = Sophus::SE3d::exp(lie);
        Sophus::SE3d delta_T = Sophus::SE3d::exp(delta_lie);

        // 李代数左乘更新
        Eigen::Matrix<double, 6, 1> x_plus_delta_lie = (delta_T * T).log();

        for(int i = 0; i < 6; ++i)
            x_plus_delta[i] = x_plus_delta_lie(i, 0);

        return true;
    }
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const
    {
        ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
        return true;
    }
    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize() const { return 6; }
};

class PnPSE3ReprojectionError : public ceres::SizedCostFunction<2, 6>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PnPSE3ReprojectionError(Eigen::Vector2d pts_2d, Eigen::Vector3d pts_3d) : _pts_2d(pts_2d), _pts_3d(pts_3d) {}

  virtual ~PnPSE3ReprojectionError() {}

  virtual bool Evaluate(
      double const *const *parameters, double *residuals, double **jacobians) const
  {

    Eigen::Map<const Eigen::Matrix<double, 6, 1>> se3(*parameters);

    Sophus::SE3d T = Sophus::SE3d::exp(se3);

    Eigen::Vector3d Pc = T * _pts_3d;

    Eigen::Matrix3d K;
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
    K << fx, 0, cx,
        0, fy, cy,
        0, 0, 1;

    Eigen::Vector2d error = _pts_2d - (K * Pc).hnormalized();

    residuals[0] = error[0];
    residuals[1] = error[1];

    if (jacobians != NULL)
    {
      if (jacobians[0] != NULL)
      {
        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(jacobians[0]);

        double x = Pc[0];
        double y = Pc[1];
        double z = Pc[2];

        double x2 = x * x;
        double y2 = y * y;
        double z2 = z * z;

        //雅克比矩阵推导看书187页公式(7.46)
        J(0, 0) = -fx / z;
        J(0, 1) = 0;
        J(0, 2) = fx * x / z2;
        J(0, 3) = fx * x * y / z2;
        J(0, 4) = -fx - fx * x2 / z2;
        J(0, 5) = fx * y / z;
        J(1, 0) = 0;
        J(1, 1) = -fy / z;
        J(1, 2) = fy * y / z2;
        J(1, 3) = fy + fy * y2 / z2;
        J(1, 4) = -fy * x * y / z2;
        J(1, 5) = -fy * x / z;
      }
    }

    return true;
  }

private:
  const Eigen::Vector2d _pts_2d;
  const Eigen::Vector3d _pts_3d;
};

//仅优化位姿
struct cost_function_define
{
  cost_function_define(Point3f p1, Point2f p2) : _p1(p1), _p2(p2) {}
  template <typename T>
  bool operator()(const T *const cere_r, const T *const cere_t, T *residual) const
  {
    T p_1[3];
    T p_2[3];
    p_1[0] = T(_p1.x);
    p_1[1] = T(_p1.y);
    p_1[2] = T(_p1.z);
    // cout << "point_3d: " << p_1[0] << " " << p_1[1] << "  " << p_1[2] << endl;
    AngleAxisRotatePoint(cere_r, p_1, p_2);
    // cout << "cam_t:" << cere_t[0] << "  " << cere_t[1] << "  " << cere_t[2] << endl;
    p_2[0] = p_2[0] + cere_t[0];
    p_2[1] = p_2[1] + cere_t[1];
    p_2[2] = p_2[2] + cere_t[2];
    // cout << "point_3d _next_2: " << p_2[0] << " " << p_2[1] << "  " << p_2[2] << endl;

    //归一化平面
    const T x = p_2[0] / p_2[2];
    const T y = p_2[1] / p_2[2];
    //三维点重投影计算的像素坐标
    //投影到像素平面
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
    const T u = x * fx + cx;
    const T v = y * fy + cy;

    //观测的在图像坐标下的值
    T u1 = T(_p2.x);
    T v1 = T(_p2.y);

    residual[0] = u1 - u;
    residual[1] = v1 - v;
    return true;
  }
  Point3f _p1;
  Point2f _p2;
};
//自动求导，优化位姿和地图点
class ReprojectionError_se3
{
public:
  ReprojectionError_se3(Eigen::Vector2d pts_2d) : _pts_2d(pts_2d) {}

  template <typename T>
  bool operator()(const T *const camera,
                  const T *const point,
                  T *residuals) const
  {
    Eigen::Map<const Eigen::Matrix<T, 6, 1>> se3(camera);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> _pts_3d(point);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> error{ residuals };
    //Eigen::Matrix<T, 6, 1> se3;
    //Eigen::Matrix<T, 3, 1> _pts_3d;
    // camera[0,1,2] are the angle-axis rotation

    //cout << "测试读取李代数数据" << camera[0]<<"2222"<<camera[1]<<camera[2];
    //cout << "测试：！！！！----se3:   " << se3.transpose() << endl;
    T predictions[2];
    
    Sophus::SE3<T> T_pose(Sophus::SE3<T>::exp(se3));
    Eigen::Matrix<T,3,1>  Pc = T_pose * _pts_3d;
    Eigen::Matrix<double,3,1> pc_test;
    Eigen::Matrix<double, 3, 3> K;
    K.template cast<T>();
    //cout << "3d点：" << _pts_3d << endl;
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
    K << fx, 0, cx,
        0, fy, cy,
        0, 0, 1;
    //auto K_new=K.cast<double>();
     error = _pts_2d - (K * Pc).hnormalized();
    return true;
  }

  static ceres::CostFunction *Create(Eigen::Vector2d pts_2d)
  {
    return (new ceres::AutoDiffCostFunction<ReprojectionError_se3, 2, 6, 3>(
        new ReprojectionError_se3(pts_2d)));
  }

private:
  const Eigen::Vector2d _pts_2d;
};

class ReprojectionError
{
public:
  ReprojectionError(double observation_x, double observation_y) : observed_x(observation_x),
                                                                  observed_y(observation_y) {}

  template <typename T>
  bool operator()(const T *const camera,
                  const T *const point,
                  T *residuals) const
  {
    // camera[0,1,2] are the angle-axis rotation
    T predictions[2];
    CamProjectionWithDistortion(camera, point, predictions);

    residuals[0] = predictions[0] - T(observed_x);
    residuals[1] = predictions[1] - T(observed_y);

    return true;
  }

  // camera : 9 dims array
  // [0-2] : angle-axis rotation //旋转
  // [3-5] : translateion        //平移
  // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion 相机焦距+去畸变参数
  // point : 3D location.
  // predictions : 2D predictions with center of the image plane.//图像平面的预测值
  template <typename T>
  static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions)
  {
    // Rodrigues' formula
    T p[3];
    //这里又有一个函数
    AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center fo distortion
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    // Apply second and fourth order radial distortion
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
    predictions[0] = xp * fx + cx;
    predictions[1] = yp * fy + cy;

    return true;
  }

  static ceres::CostFunction *Create(const double observed_x, const double observed_y)
  {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
        new ReprojectionError(observed_x, observed_y)));
  }

private:
  double observed_x;
  double observed_y;
};

//手动求导，优化位姿和地图点
class SE3ReprojectionError : public ceres::SizedCostFunction<2, 6, 3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SE3ReprojectionError(Eigen::Vector2d pts_2d) : _pts_2d(pts_2d) {}

  virtual ~SE3ReprojectionError() {}

  virtual bool Evaluate(
      double const *const *parameters, double *residuals, double **jacobians) const
  {

    Eigen::Map<const Eigen::Matrix<double, 6, 1>> se3(parameters[0]);
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> _pts_3d(parameters[1]);
    Sophus::SE3d T(Sophus::SE3d::exp(se3));
    Eigen::Vector3d Pc = T * _pts_3d;
    Eigen::Matrix3d K;

    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
    K << fx, 0, cx,
        0, fy, cy,
        0, 0, 1;

    Eigen::Vector2d error = _pts_2d - (K * Pc).hnormalized();
    residuals[0] = error[0];
    residuals[1] = error[1];

    if (jacobians != NULL)
    {
      if (jacobians[0] != NULL)
      {
        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(jacobians[0]);

        double x = Pc[0];
        double y = Pc[1];
        double z = Pc[2];

        double x2 = x * x;
        double y2 = y * y;
        double z2 = z * z;

        //误差关于扰动李代数导数
        J(0, 0) = -fx / z;
        J(0, 1) = 0;
        J(0, 2) = fx * x / z2;
        J(0, 3) = fx * x * y / z2;
        J(0, 4) = -fx - fx * x2 / z2;
        J(0, 5) = fx * y / z;

        J(1, 0) = 0;
        J(1, 1) = -fy / z;
        J(1, 2) = fy * y / z2;
        J(1, 3) = fy + fy * y2 / z2;
        J(1, 4) = -fy * x * y / z2;
        J(1, 5) = -fy * x / z;
      }
      if (jacobians[1] != NULL)
      {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J2(jacobians[1]);
        Eigen::Matrix<double,2,3> J_first;
        double x = Pc[0];
        double y = Pc[1];
        double z = Pc[2];

        double x2 = x * x;
        double y2 = y * y;
        double z2 = z * z;

        //关于3d点导数
        J_first(0, 0) = fx / z;
        J_first(0, 1) = 0;
        J_first(0, 2) = -fx * x / z2;

        //关于3d点导数
        J_first(1, 0) = 0;
        J_first(1, 1) = fy / z;
        J_first(1, 2) = -fy * y / z2;
        J2=-J_first*T.rotationMatrix();
      }
    }

    return true;
  }

private:
  const Eigen::Vector2d _pts_2d;
};

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
  Eigen::Matrix3d K;
  K << 458.654, 0, 367.215,
      0, 457.296, 248.375,
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
  correctMatches(F_opencv, points_1, points_2, points_1_new, points_2_new);
  
  cv::Mat E_opencv = K_cv.t() * F_opencv * K_cv;
  cout << "F_opencv\n"
       << F_opencv << endl;
  cv::Mat R_cv_estimated, t_cv_estimated;
  cv::Mat E_opencv2 = findEssentialMat(points_1_new, points_2_new, K_cv);
  recoverPose(E_opencv, points_1_new, points_2_new, K_cv, R_cv_estimated, t_cv_estimated);

  // cout << "E_opencv\n"
  //      << E_opencv << endl;
  // cout << "E_opencv2\n"
  //      << E_opencv2 << endl;
  // cout << "R\n"
  //      << R_cv_estimated << endl;
  // cout << "t\n"
  //      << t_cv_estimated << endl;

  vector<cv::Point3d> pts_3d;
  triangulation_opencv(keypoints_1, keypoints_2, matches_after_ransac, R_cv_estimated, t_cv_estimated, pts_3d, K_cv);
  //优化之前的重投影误差计算
  //
  double error_before_BA = 0;
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
    //cout << "第" << i << "个点的重投影误差" << error_point.norm() << endl;
    error_before_BA += error_point.norm();
  }

  float mean_error_before_BA = error_before_BA / pts_3d.size() * 1.0;
  cout << "error_before_BA: " << error_before_BA << "  mean_error_before_BA" << mean_error_before_BA << endl;
  
  vector<double> r_vector, t_vector(3);
  double cere_rot[3], cere_tranf[3];
  cv::Rodrigues(R_cv_estimated, r_vector);
  t_vector[0] = t_cv_estimated.at<double>(0, 0);
  t_vector[1] = t_cv_estimated.at<double>(1, 0);
  t_vector[2] = t_cv_estimated.at<double>(2, 0);
  double camera[6];
  camera[0] = r_vector[0];
  camera[1] = r_vector[1];
  camera[2] = r_vector[2];
  camera[3] = t_vector[0];
  camera[4] = t_vector[1];
  camera[5] = t_vector[2];
  //定义3D点数组
  double *point = new double[pts_3d.size() * 3];
  vector<Point2f> &pts_2d = points_2_new;

  Eigen::Matrix3d R_eigen;
  Eigen::Vector3d t_eigen;
  cv2eigen(R_cv_estimated, R_eigen);
  cv2eigen(t_cv_estimated, t_eigen);
  //构建李代数
  Sophus::SE3d T_se(R_eigen, t_eigen);
  Eigen::Matrix<double, 6, 1> se3(T_se.log());

  vector<Eigen::Vector3d> pts_3d_eigen;
  vector<Eigen::Vector2d> pts_2d_eigen;
  for (int i = 0; i < pts_2d.size(); i++)
  {
    pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
  }

  auto pts_3d_eigen_test = pts_3d_eigen;
  auto se3_test = se3;
  int N_test = pts_2d_eigen.size();
  ceres::Problem problem_auto_posepoint;
  for (int i = 0; i < N_test; ++i)
  {
    ceres::CostFunction *cost_function;
    cost_function = ReprojectionError_se3::Create(pts_2d_eigen[i]);
    // ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    problem_auto_posepoint.AddResidualBlock(cost_function, NULL, se3_test.data(), pts_3d_eigen_test[i].data());
  }

  ceres::Solver::Options options_test;
  ceres::Solver::Summary summary_test;
  options_test.dynamic_sparsity = true;
  options_test.max_num_iterations = 100;
  options_test.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options_test.minimizer_type = ceres::TRUST_REGION;
  options_test.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options_test.trust_region_strategy_type = ceres::DOGLEG;
  options_test.minimizer_progress_to_stdout = true;
  options_test.dogleg_type = ceres::SUBSPACE_DOGLEG;
  //梯度检查
  //options_test.check_gradients = ;

  auto t1_auto_test = chrono::steady_clock::now();
  ceres::Solve(options_test, &problem_auto_posepoint, &summary_test);
  std::cout << summary_test.BriefReport() << "\n";
  auto t2_auto_test = chrono::steady_clock::now();
  auto time_auto_used_test = chrono::duration_cast<chrono::duration<double>>(t2_auto_test - t1_auto_test);
  cout << "ceres自动求导耗时(数据格式se3+3D点): " << time_auto_used_test.count() << " seconds." << endl;

  vector<double> verror_auto_pointpose_BA;
  double error_auto_pointpose_BA = 0;
  for (int i = 0; i < pts_3d_eigen_test.size(); i++)
  {
    Sophus::SE3d T(Sophus::SE3d::exp(se3_test));
    Eigen::Vector3d Pc = T * pts_3d_eigen[i];

    Eigen::Vector2d error = pts_2d_eigen[i] - (K * Pc).hnormalized();

    cout << "第" << i << "个点的重投影误差" << error.norm() << endl;
    error_auto_pointpose_BA += error.norm();
    verror_auto_pointpose_BA.push_back(error.norm());
  }
  cout << "平均投影误差" << error_auto_pointpose_BA / pts_3d_eigen_test.size() << endl;

  //*自动求导
  //*仅优化位姿

  // //*数据格式定义
  // cere_rot[0] = r_vector[0];
  // cere_rot[1] = r_vector[1];
  // cere_rot[2] = r_vector[2];
  // cere_tranf[0] = t_vector[0];
  // cere_tranf[1] = t_vector[1];
  // cere_tranf[2] = t_vector[2];
  // ceres::Problem problem2;
  // for (int i = 0; i < pts_3d.size(); i++)
  // {
  //   ceres::CostFunction *costfunction = new ceres::AutoDiffCostFunction<cost_function_define, 2, 3, 3>(new cost_function_define(pts_3d[i], pts_2d[i]));
  //   ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
  //   problem2.AddResidualBlock(costfunction, loss_function, cere_rot, cere_tranf); //注意，cere_rot不能为Mat类型
  // }
  // ceres::Solver::Options option2;
  // option2.linear_solver_type = ceres::DENSE_SCHUR;
  // //输出迭代信息到屏幕
  // // option.minimizer_progress_to_stdout = true;
  // //显示优化信息
  // ceres::Solver::Summary summary2;
  // //开始求解
  // ceres::Solve(option2, &problem2, &summary2);
  // //显示优化信息
  // // cout << summary.BriefReport() << endl;
  // // cout << "仅优化位姿"
  // //      << "-----------------" << endl;
  // // cout << "优化之前r：" << r_vector[0] << " " << r_vector[1] << " " << r_vector[2] << endl;
  // // cout << "优化之后r: " << cere_rot[0] << "  " << cere_rot[1] << "  " << cere_rot[2] << endl;
  // // cout << "优化之前t：" << t_cv_estimated.at<double>(0, 0) << "  " << t_cv_estimated.at<double>(1, 0) << "  " << t_cv_estimated.at<double>(2, 0) << endl;
  // // cout << "优化之后t: " << cere_tranf[0] << "  " << cere_tranf[1] << "  " << cere_tranf[2] << endl;

  // //*重投影误差计算
  // double error_onlypose_BA = 0;
  // for (int i = 0; i < pts_3d.size(); i++)
  // {
  //   Mat P3D = (Mat_<double>(3, 1) << pts_3d[i].x, pts_3d[i].y, pts_3d[i].z);
  //   Mat r = (Mat_<double>(3, 1) << cere_rot[0], cere_rot[1], cere_rot[2]);
  //   Mat t = (Mat_<double>(3, 1) << cere_tranf[0], cere_tranf[1], cere_tranf[2]);
  //   Mat R;
  //   Rodrigues(r, R);

  //   Mat pts_3d_img2 = R * P3D + t;
  //   double xp = pts_3d_img2.at<double>(0) / pts_3d_img2.at<double>(2);
  //   double yp = pts_3d_img2.at<double>(1) / pts_3d_img2.at<double>(2);

  //   double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
  //   double u = xp * fx + cx;
  //   double v = yp * fy + cy;
  //   double dx = points_2[i].x - u;
  //   double dy = points_2[i].y - v;

  //   Eigen::Vector2d error_point(dx, dy);
  //   cout << "第" << i << "个点的重投影误差" << error_point.norm() << endl;
  //   error_onlypose_BA += error_point.norm();
  // }
  // cout << "平均重投影误差：" << error_onlypose_BA / pts_3d.size() << endl;

  // //*优化位姿和地图点 旋转向量+平移向量+位姿

  // ceres::Problem problem;
  // for (int i = 0; i < pts_2d.size(); ++i)
  // {
  //   ceres::CostFunction *cost_function;
  //   cost_function = ReprojectionError::Create(pts_2d[i].x, pts_2d[i].y);
  //   // ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
  //   point[i * 3] = pts_3d[i].x;
  //   point[i * 3 + 1] = pts_3d[i].y;
  //   point[i * 3 + 2] = pts_3d[i].z;
  //   //! bug出现在这里，每次都传的是首地址，每次都优化一个点，这怎么能优化得了
  //   problem.AddResidualBlock(cost_function, NULL, camera, point + 3 * i);
  // }

  // auto t1_auto = chrono::steady_clock::now();
  // std::cout << "Solving ceres BA ... " << endl;
  // ceres::Solver::Options options;
  // options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
  // options.minimizer_progress_to_stdout = true;
  // ceres::Solver::Summary summary;
  // ceres::Solve(options, &problem, &summary);

  // auto t2_auto = chrono::steady_clock::now();
  // auto time_auto_used = chrono::duration_cast<chrono::duration<double>>(t2_auto - t1_auto);
  // cout << "ceres自动求导: " << time_auto_used.count() << " seconds." << endl;
  // // cout << "优化之前r：" << r_vector[0] << " " << r_vector[1] << " " << r_vector[2] << endl;
  // // cout << "优化之后r: " << camera[0] << "  " << camera[1] << "  " << camera[2] << endl;
  // // cout << "优化之前t：" << t_cv_estimated.at<double>(0, 0) << "  " << t_cv_estimated.at<double>(1, 0) << "  " << t_cv_estimated.at<double>(2, 0) << endl;
  // // cout << "优化之后t: " << camera[3] << "  " << camera[4] << "  " << camera[5] << endl;

  // //计算重投影误差
  // int N = pts_3d.size();
  // double reprejection_error = 0, mean_error = 0, sqrt_error = 0;
  // //保存每个点的重投影误差
  // vector<double> error_auto_posepoint_ba(N, 0);
  // for (int i = 0; i < N; i++)
  // {
  //   double x, y, z;
  //   x = point[i * 3];
  //   y = point[i * 3 + 1];
  //   z = point[i * 3 + 2];

  //   Mat P3D_c1 = (Mat_<double>(3, 1) << x, y, z);
  //   Mat r = (Mat_<double>(3, 1) << camera[0], camera[1], camera[2]);
  //   Mat t_cv_afterba_estimated = (Mat_<double>(3, 1) << camera[3], camera[4], camera[5]);
  //   Mat R_cv_afterba_estimated;
  //   Rodrigues(r, R_cv_afterba_estimated);
  //   Mat P3D_c2 = R_cv_afterba_estimated * P3D_c1 + t_cv_afterba_estimated;

  //   double xp = P3D_c2.at<double>(0) / P3D_c2.at<double>(2);
  //   double yp = P3D_c2.at<double>(1) / P3D_c2.at<double>(2);

  //   double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
  //   double u = xp * fx + cx;
  //   double v = yp * fy + cy;
  //   double dx = pts_2d[i].x - u;
  //   double dy = pts_2d[i].y - v;

  //   Eigen::Vector2d error_point(dx, dy);
  //   error_auto_posepoint_ba[i] = error_point.norm();
  //   cout << "第" << i << "个点的重投影误差" << error_auto_posepoint_ba[i] << endl;
  //   reprejection_error += error_auto_posepoint_ba[i];
  // }
  // //计算方差和标准差
  // cout << "平均重投影误差：" << reprejection_error / N << endl;


  //*手动求导
   ceres::Problem problem_;
   ceres::LocalParameterization *local_param = new SE3Parameterization();
   problem_.AddParameterBlock(se3.data(), 6, local_param);
  for (int i = 0; i < pts_2d_eigen.size(); ++i)
  {
    ceres::CostFunction *cost_function;
    cost_function = new SE3ReprojectionError(pts_2d_eigen[i]);
    problem_.AddResidualBlock(cost_function, NULL, se3.data(), pts_3d_eigen[i].data());
    
  }

  
  // for(int i=0; i<pts_3d_eigen.size(); ++i) {
  //     ceres::CostFunction *cost_function;
  //     cost_function = new PnPSE3ReprojectionError(pts_2d_eigen[i], pts_3d_eigen[i]);
  //     problem_.AddResidualBlock(cost_function, NULL, se3.data());
  // }

  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;
  options_.dynamic_sparsity = true;
  options_.max_num_iterations = 100;
  options_.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options_.minimizer_type = ceres::TRUST_REGION;
  options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options_.trust_region_strategy_type = ceres::DOGLEG;
  options_.minimizer_progress_to_stdout = true;
  options_.dogleg_type = ceres::SUBSPACE_DOGLEG;
  //梯度检查
  options_.check_gradients = true;

  auto t1_manual = chrono::steady_clock::now();
  ceres::Solve(options_, &problem_, &summary_);
  std::cout << summary_.BriefReport() << "\n";
  auto t2_manual = chrono::steady_clock::now();
  auto time_used = chrono::duration_cast<chrono::duration<double>>(t2_manual - t1_manual);
  cout << "ceres手动求导耗时: " << time_used.count() << " seconds." << endl;

  //*重投影误差计算
  vector<double> verror_manunal_pointpose_BA;
  double error_manunal_pointpose_BA = 0;
  for (int i = 0; i < pts_3d_eigen.size(); i++)
  {
    Sophus::SE3d T(Sophus::SE3d::exp(se3));
    Eigen::Vector3d Pc = T * pts_3d_eigen[i];

    Eigen::Vector2d error = pts_2d_eigen[i] - (K * Pc).hnormalized();

    cout << "第" << i << "个点的重投影误差" << error.norm() << endl;
    error_manunal_pointpose_BA += error.norm();
    verror_manunal_pointpose_BA.push_back(error.norm());
  }
  cout << "平均投影误差" << error_manunal_pointpose_BA / pts_3d_eigen.size() << endl;

  // std::cout << "estimated pose: \n"
  //           << Sophus::SE3d::exp(se3).matrix() << std::endl;
  return 0;
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
