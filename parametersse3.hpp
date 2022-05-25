#ifndef PARAMETERSSE3_HPP
#define PARAMETERSSE3_HPP

#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <iostream>

#include <ceres/ceres.h>
#include "rotation.h"
#include "se3.hpp"

using namespace std;
using namespace Eigen;

class CameraParameters
{
protected:
    double f;
    double cx;
    double cy;
public:
    CameraParameters(double f_, double cx_, double cy_)
        : f(f_), cx(cx_), cy(cy_) {}

    Vector2d cam_map(const Vector3d& p)
    {
        Vector2d z;
        z[0] = f * p[0] / p[2] + cx;
        z[1] = f * p[1] / p[2] + cy;
        return z;
    }
};

/// PoseBlockSize can only be
/// 7 (quaternion + translation vector) or
/// 6 (rotation vector + translation vector)
template<int PoseBlockSize>
class ReprojectionErrorSE3XYZ: public ceres::SizedCostFunction<2, PoseBlockSize, 3>
{
public:
    ReprojectionErrorSE3XYZ(double f_,
                            double cx_,
                            double cy_,
                            double observation_x,
                            double observation_y)
        : f(f_), cx(cx_), cy(cy_),
          _observation_x(observation_x),
          _observation_y(observation_y){}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    double f;
    double cx;
    double cy;

private:
    double _observation_x;
    double _observation_y;
};

/// PoseBlockSize can only be
/// 7 (quaternion + translation vector) or
/// 6 (rotation vector + translation vector)
template<int PoseBlockSize>
class PoseSE3Parameterization : public ceres::LocalParameterization {
public:
    PoseSE3Parameterization() {}
    virtual ~PoseSE3Parameterization() {}
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const;
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const;
    virtual int GlobalSize() const { return PoseBlockSize; }
    virtual int LocalSize() const { return 6; }
};

/// PoseBlockSize can only be
/// 7 (quaternion + translation vector) or
/// 6 (rotation vector + translation vector)
template<int PoseBlockSize>
class PosePointParametersBlock
{
public:
    PosePointParametersBlock(){}
    void create(int pose_num, int point_num)
    {
        poseNum = pose_num;
        pointNum = point_num;
        values = new double[pose_num * PoseBlockSize + point_num * 3];
    }
    PosePointParametersBlock(int pose_num, int point_num): poseNum(pose_num), pointNum(point_num)
    {
        values = new double[pose_num * PoseBlockSize + point_num * 3];
    }
    ~PosePointParametersBlock() { delete[] values; }

    void setPose(int idx, const Quaterniond &q, const Vector3d &trans);

    void getPose(int idx, Quaterniond &q, Vector3d &trans);

    double* pose(int idx) {  return values + idx * PoseBlockSize; }

    double* point(int idx) { return values + poseNum * PoseBlockSize + idx * 3; }

    int poseNum;
    int pointNum;
    double *values;

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

  template <typename T>
  static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions)
  {
    // Rodrigues' formula
    T p[3];
    //旋转
    AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center fo distortion
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    double fx = 458.654, fy = 458.654, cx = 367.215, cy = 248.375;
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


class ReprojectionErrorse3: public ceres::SizedCostFunction<2, 6, 3>
{
public:
    ReprojectionErrorse3(double f_,
                            double cx_,
                            double cy_,
                            double observation_x,
                            double observation_y)
        : f(f_), cx(cx_), cy(cy_),
          _observation_x(observation_x),
          _observation_y(observation_y){}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    double f;
    double cx;
    double cy;

private:
    double _observation_x;
    double _observation_y;
};

class Pose_se3Parameterization : public ceres::LocalParameterization {
public:
    Pose_se3Parameterization() {}
    virtual ~Pose_se3Parameterization() {}
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const;
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const;
    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize() const { return 6; }
};

#endif // PARAMETERSSE3_HPP
