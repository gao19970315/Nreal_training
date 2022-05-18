#include <iostream>

using namespace std;

#include <ctime>
// Eigen 核心部分
#include <Eigen/Core>
// 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Dense>

using namespace std;

int main(int argc,char ** argv){
    Eigen::Vector3d t_l(-0.050720060477640147,-0.0017414170413474165,0.0022943667597148118);
    Eigen::Vector3d t_r(0.051932496584961352,-0.0011555929083120534,0.0030949732069645722);

    Eigen::Quaterniond r_l(0.99090224973327068,0.13431639597354814,0.00095051670014565813,-0.0084222184858180373);
    Eigen::Quaterniond r_r(0.99073762672679389,0.13492462817073628,-0.00013648999867379373,-0.015306242884176362);

    Eigen::Isometry3d T_l = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_r = Eigen::Isometry3d::Identity();
    T_l.rotate(r_l);
    T_l.pretranslate(t_l);
    
    T_r.rotate(r_r);
    T_r.pretranslate(t_r);

    auto T=T_l.inverse()*T_r;
    Eigen::Quaterniond right_q_left (T.rotation());
    Eigen::Vector3d right_p_left (T.translation());
    
    cout <<right_p_left.transpose()<<endl;
    cout <<right_q_left.coeffs().transpose()<<endl;

}