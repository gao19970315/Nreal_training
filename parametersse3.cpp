#include "sophus/se3.hpp"
#include "parametersse3.hpp"

template <>
bool ReprojectionErrorSE3XYZ<7>::Evaluate(const double *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Map<const Quaterniond> quaterd(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> trans(parameters[0] + 4);
    Eigen::Map<const Eigen::Vector3d> point(parameters[1]);

    Eigen::Vector3d p = quaterd * point + trans;

    double f_by_z = f / p[2];
    residuals[0] = f_by_z * p[0] + cx - _observation_x;
    residuals[1] = f_by_z * p[1] + cy - _observation_y;

    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_cam;
    double f_by_zz = f_by_z / p[2];
    J_cam << f_by_z, 0, -f_by_zz * p[0],
        0, f_by_z, -f_by_zz * p[1];

    if (jacobians != NULL)
    {
        if (jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_se3(jacobians[0]);
            J_se3.setZero();
            J_se3.block<2, 3>(0, 0) = -J_cam * skew(p);
            J_se3.block<2, 3>(0, 3) = J_cam;
        }
        if (jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_point(jacobians[1]);
            J_point = J_cam * quaterd.toRotationMatrix();
        }
    }
    return true;
}

template <>
bool ReprojectionErrorSE3XYZ<6>::Evaluate(const double *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Quaterniond quaterd = toQuaterniond(Eigen::Map<const Vector3d>(parameters[0]));
    Eigen::Map<const Eigen::Vector3d> trans(parameters[0] + 3);
    Eigen::Map<const Eigen::Vector3d> point(parameters[1]);

    Eigen::Vector3d p = quaterd * point + trans;

    double f_by_z = f / p[2];
    residuals[0] = f_by_z * p[0] + cx - _observation_x;
    residuals[1] = f_by_z * p[1] + cy - _observation_y;

    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_cam;
    double f_by_zz = f_by_z / p[2];
    J_cam << f_by_z, 0, -f_by_zz * p[0],
        0, f_by_z, -f_by_zz * p[1];

    if (jacobians != NULL)
    {
        if (jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_se3(jacobians[0]);
            J_se3.block<2, 3>(0, 0) = -J_cam * skew(p); // p的反对称矩阵
            J_se3.block<2, 3>(0, 3) = J_cam;
        }
        if (jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_point(jacobians[1]);
            J_point = J_cam * quaterd.toRotationMatrix();
        }
    }

    return true;
}

template <>
bool PoseSE3Parameterization<7>::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> trans(x + 4);
    SE3 se3_delta = SE3::exp(Eigen::Map<const Vector6d>(delta));

    Eigen::Map<const Eigen::Quaterniond> quaterd(x);
    Eigen::Map<Eigen::Quaterniond> quaterd_plus(x_plus_delta);
    Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 4);

    quaterd_plus = se3_delta.rotation() * quaterd;
    trans_plus = se3_delta.rotation() * trans + se3_delta.translation();

    return true;
}

template <>
bool PoseSE3Parameterization<7>::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
    J.setZero();
    J.block<6, 6>(0, 0).setIdentity();
    return true;
}

template <>
bool PoseSE3Parameterization<6>::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> trans(x + 3);
    SE3 se3_delta = SE3::exp(Eigen::Map<const Vector6d>(delta));

    Quaterniond quaterd_plus = se3_delta.rotation() * toQuaterniond(Eigen::Map<const Vector3d>(x));
    Eigen::Map<Vector3d> angles_plus(x_plus_delta);
    angles_plus = toAngleAxis(quaterd_plus);

    Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 3);
    trans_plus = se3_delta.rotation() * trans + se3_delta.translation();
    return true;
}

template <>
bool PoseSE3Parameterization<6>::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
    J.setIdentity();
    return true;
}

bool ReprojectionErrorse3::Evaluate(double const *const *parameters,
                                    double *residuals,
                                    double **jacobians) const
{
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> se3(parameters[0]);
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> _pts_3d(parameters[1]);
    Sophus::SE3d T(Sophus::SE3d::exp(se3));
    Eigen::Vector3d p = T * _pts_3d;

    double f_by_z = f / p[2];
    residuals[0] = f_by_z * p[0] + cx - _observation_x;
    residuals[1] = f_by_z * p[1] + cy - _observation_y;

    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_cam;
    double f_by_zz = f_by_z / p[2];
    J_cam << f_by_z, 0, -f_by_zz * p[0],
        0, f_by_z, -f_by_zz * p[1];

    if (jacobians != NULL)
    {
        if (jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_se3(jacobians[0]);
            Eigen::Matrix<double, 2, 6> J_se3_1;
            J_se3_1.block<2, 3>(0, 0) = J_cam; // p的反对称矩阵
            J_se3_1.block<2, 3>(0, 3) = -J_cam * skew(p);


            // se3上的平移量
            Eigen::Vector3d p_ = se3.topRows(3);

            //转轴，方向向量
            Eigen::Vector3d sita = Eigen::AngleAxisd(T.rotationMatrix()).axis();

            //角度即模长
            double sita_norm = Eigen::AngleAxisd(T.rotationMatrix()).angle();
            // cout<<"se3"<<se3.matrix().transpose()<<endl;
            // cout<<"se3上的位移量p"<<p_<<endl;
            // cout<<"方向向量"<<sita.transpose()<<"方向向量模长"<<sita.norm()<<"模长"<<sita_norm<<endl;
            //恢复旋转向量
            sita = sita * sita_norm;

            double sita_norm2 = sita_norm * sita_norm;
            double sita_norm3 = sita_norm2 * sita_norm;
            double sita_norm4 = sita_norm3 * sita_norm;
            double sita_norm5 = sita_norm4 * sita_norm;

            Eigen::Matrix3d sita_hat, sita_hat2, p_hat;
            sita_hat = skew(sita);
            sita_hat2 = sita_hat * sita_hat;
            p_hat = skew(p_);

            Eigen::Matrix3d J_l_sita, Q_p;
            Eigen::Matrix3d matrix3d_identity = Eigen::MatrixXd::Identity(3, 3);

            J_l_sita = matrix3d_identity + (1 - cos(sita_norm)) / (sita_norm2)*sita_hat + (sita_norm - sin(sita_norm)) / sita_norm3 * sita_hat2;
            Q_p = 0.5 * p_hat + (sita_norm - sin(sita_norm)) / sita_norm3 * (sita_hat * p_hat + p_hat * sita_hat + sita_hat * p_hat * sita_hat) - (1 - 0.5 * sita_norm2 - cos(sita_norm)) / sita_norm4 * (sita_hat2 * p_hat + p_hat * sita_hat2 - 3 * sita_hat * p_hat * sita_hat) 
            - 0.5 * ((1 - 0.5 * sita_norm2 - cos(sita_norm)) / sita_norm4 - 3 * (sita_norm - sin(sita_norm) - sita_norm3 / 6) / sita_norm5) * (sita_hat * p_hat * sita_hat2 + sita_hat2 * p_hat * sita_hat);
            Eigen::Matrix<double, 6, 6> J_se3_2;
            J_se3_2.setZero();
            J_se3_2.topLeftCorner(3, 3) = J_l_sita;
            J_se3_2.topRightCorner(3, 3) = Q_p;
            J_se3_2.bottomRightCorner(3, 3) = J_l_sita;

            J_se3 = J_se3_1 * J_se3_2;
            // cout<<"j_se3_2"<<J_se3_2<<endl;
        }
        if (jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_point(jacobians[1]);
            J_point = J_cam * T.rotationMatrix();
        }
    }
    return true;
}

bool Pose_se3Parameterization::Plus(const double *x,
                                    const double *delta,
                                    double *x_plus_delta) const 
{
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(x);
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_lie(delta);
    Eigen::Map<Eigen::Matrix<double, 6, 1>> update_lie(x_plus_delta);
    update_lie = lie + delta_lie;
    return true;
}
bool Pose_se3Parameterization::ComputeJacobian(const double *x,
                                               double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
    J.setIdentity();
    return true;
}

template <>
void PosePointParametersBlock<7>::getPose(int idx, Quaterniond &q, Vector3d &trans)
{
    double *pose_ptr = values + idx * 7;
    q = Map<const Quaterniond>(pose_ptr);
    trans = Map<const Vector3d>(pose_ptr + 4);
}

template <>
void PosePointParametersBlock<7>::setPose(int idx, const Quaterniond &q, const Vector3d &trans)
{
    double *pose_ptr = values + idx * 7;
    Eigen::Map<Vector7d> pose(pose_ptr);
    pose.head<4>() = Eigen::Vector4d(q.coeffs());
    pose.tail<3>() = trans;
}

template <>
void PosePointParametersBlock<6>::getPose(int idx, Quaterniond &q, Vector3d &trans)
{
    double *pose_ptr = values + idx * 6;
    q = toQuaterniond(Vector3d(pose_ptr));
    trans = Map<const Vector3d>(pose_ptr + 3);
}

template <>
void PosePointParametersBlock<6>::setPose(int idx, const Quaterniond &q, const Vector3d &trans)
{
    double *pose_ptr = values + idx * 6;
    Eigen::Map<Vector6d> pose(pose_ptr);
    pose.head<3>() = toAngleAxis(q);
    pose.tail<3>() = trans;
}
