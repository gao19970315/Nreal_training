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
/**
 * Custom Edge and Vertex 
 */
struct myEdge
{
  int xi,xj;    // 对应节点的索引
  Eigen::Vector3d measurement;  // 两个节点之间的相对位姿的观测值，
                                // 分别为：dx, dy, dtheta
  Eigen::Matrix3d infoMatrix;   // 本次观测的信息矩阵
};

typedef Eigen::Vector3d myVertex;   // 机器人位姿，x, y, theta

class AutoDiffFunctor {
public:
    AutoDiffFunctor(const myEdge& edge): measurement(edge.measurement),
                                 sqrt_info_matrix(edge.infoMatrix.array().sqrt())
    {}

    template <typename T>
    bool operator()(const T* const v1, const T* const v2, T* residual) const {
        
        
        Eigen::Matrix<T, 3, 1> v1_mat{ v1[0], v1[1], v1[2] };
        Eigen::Matrix<T, 3, 1> v2_mat{ v2[0], v2[1], v2[2] };
        Eigen::Matrix<T, 3, 1> m = measurement.template cast<T>();
        Eigen::Map<Eigen::Matrix<T, 3, 1>> error{ residual };

        // calculate error from translation and 
        // rotation respectively and combine them together
        Eigen::Matrix<T, 3, 3> X1 = PoseToTrans(v1_mat);
        Eigen::Matrix<T, 3, 3> X2 = PoseToTrans(v2_mat);
        Eigen::Matrix<T, 3, 3> Z = PoseToTrans(m);
        
        Eigen::Matrix<T, 2, 2> Ri = X1.block(0, 0, 2, 2);
        Eigen::Matrix<T, 2, 2> Rj = X2.block(0, 0, 2, 2);
        Eigen::Matrix<T, 2, 2> Rij = Z.block(0, 0, 2, 2);

        Eigen::Matrix<T, 2, 1> ti{ v1_mat(0), v1_mat(1) };
        Eigen::Matrix<T, 2, 1> tj{ v2_mat(0), v2_mat(1) };
        Eigen::Matrix<T, 2, 1> tij{ m(0), m(1) };

        Eigen::Matrix<T, 2, 2> dRiT_dtheta;   //  derivative of Ri^T over theta
        dRiT_dtheta(0, 0) = T(-1) * Ri(1, 0); //  cosX -> -sinX
        dRiT_dtheta(0, 1) = T( 1) * Ri(0, 0); //  sinX ->  cosX
        dRiT_dtheta(1, 0) = T(-1) * Ri(0, 0); // -sinX -> -cosX
        dRiT_dtheta(1, 1) = T(-1) * Ri(1, 0); //  cosX -> -sinX

        // calcuate error & normalize error on theta
        error.template segment<2>(0) = \
                Rij.transpose() * (Ri.transpose() * (tj - ti) - tij);
        error(2) = v2_mat(2) - v1_mat(2) - m(2);
        if (error(2) > T(M_PI)) {
            error(2) -= T(2 * M_PI);
        } else if (error(2) < T(-1 * M_PI)) {
            error(2) += T(2 * M_PI);
        }

        error = sqrt_info_matrix.template cast<T>() * error;

        return true;
    }

    static ceres::CostFunction* create(const myEdge& edge) {
        return (new ceres::AutoDiffCostFunction<AutoDiffFunctor, 3, 3, 3>(new AutoDiffFunctor(edge)));
    }

private:
    Eigen::Vector3d measurement;
    Eigen::Matrix3d sqrt_info_matrix;
};


int main(){
    /// 获取我们需要优化的节点和边
std::vector<myVertex> Vertexs;
std::vector<myEdge> Edges;


/// 声明一个问题
ceres::Problem problem;

/// 核心部分：
for (const auto& edge: Edges) {
    // 对每一次观测，生成一个 Cost Function，也就是误差函数
    ceres::CostFunction* cost_function = AutoDiffFunctor::create(edge);
    // 将这一次观测涉及到的节点以残差块的形式添加进问题中
    problem.AddResidualBlock(cost_function, 
                            nullptr, Vertexs[edge.xi].data(), 
                            Vertexs[edge.xj].data());
}

/// 节点和边添加完成后就可以求解
ceres::Solver::Options options;              // 不同求解选项
options.linear_solver_type = \
ceres::SPARSE_NORMAL_CHOLESKY;               // 这里使用稀疏 Cholesky 求解线性方程组
options.minimizer_progress_to_stdout = true; // 输出求解过程
ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);   // 求解问题
std::cout << summary.FullReport() << "\n";   // 输出结果

}

class AnalyticDiffFunction : public ceres::SizedCostFunction<3, 3, 3> {
public:
    virtual ~AnalyticDiffFunction() {}
    
    AnalyticDiffFunction(const myEdge& edge): 
                            measurement(edge.measurement),
                            sqrt_info_matrix(edge.infoMatrix.array().sqrt()) 
                            {}
    
    virtual bool Evaluate(double const* const* parameters,
                         double* residuals, double** jacobians) const {

        Eigen::Vector3d xi{ parameters[0][0],
                            parameters[0][1],
                            parameters[0][2] };
        Eigen::Vector3d xj{ parameters[1][0],
                            parameters[1][1],
                            parameters[1][2] };

        Eigen::Map<Eigen::Vector3d> error_ij{ residuals };
        Eigen::Matrix3d Ai;
        Eigen::Matrix3d Bi;

        Eigen::Matrix3d Xi = PoseToTrans(xi);
        Eigen::Matrix2d Ri = Xi.block(0, 0, 2, 2);
        Eigen::Vector2d ti{ xi(0), xi(1) };

        Eigen::Matrix3d Xj = PoseToTrans(xj);
        Eigen::Matrix2d Rj = Xj.block(0, 0, 2, 2);
        Eigen::Vector2d tj{ xj(0), xj(1) };

        Eigen::Matrix3d Z  = PoseToTrans(measurement);
        Eigen::Matrix2d Rij = Z.block(0, 0, 2, 2);
        Eigen::Vector2d tij{ measurement(0), measurement(1) };

        Eigen::Matrix2d dRiT_dtheta;       //  derivative of Ri^T over theta
        dRiT_dtheta(0, 0) = -1 * Ri(1, 0); //  cosX -> -sinX
        dRiT_dtheta(0, 1) =  1 * Ri(0, 0); //  sinX ->  cosX
        dRiT_dtheta(1, 0) = -1 * Ri(0, 0); // -sinX -> -cosX
        dRiT_dtheta(1, 1) = -1 * Ri(1, 0); //  cosX -> -sinX

        // calcuate error & normalize error on theta
        error_ij.segment<2>(0) =\
             Rij.transpose() * (Ri.transpose() * (tj - ti) - tij);
        error_ij(2) = xj(2) - xi(2) - measurement(2);
        if (error_ij(2) > M_PI) {
            error_ij(2) -= 2 * M_PI;
        } else if (error_ij(2) < -1 * M_PI) {
            error_ij(2) += 2 * M_PI;
        }

        Ai.setZero();
        Ai.block(0, 0, 2, 2) = -Rij.transpose() * Ri.transpose();
        Ai.block(0, 2, 2, 1) = Rij.transpose() * dRiT_dtheta * (tj - ti);
        Ai(2, 2) = -1.0;

        Bi.setIdentity();
        Bi.block(0, 0, 2, 2) = Rij.transpose() * Ri.transpose();

        error_ij = sqrt_info_matrix * error_ij;

        if(jacobians){
            if(jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> \
                        jacobian_xi(jacobians[0]);
                jacobian_xi = sqrt_info_matrix * Ai;
            }
            if(jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> \
                        jacobian_xj(jacobians[1]);
                jacobian_xj = sqrt_info_matrix * Bi;
            }
        }

        return true;
    }

private:
    Eigen::Vector3d measurement;
    Eigen::Matrix3d sqrt_info_matrix;
};

