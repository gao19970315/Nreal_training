#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "common/rotation.h"
using namespace std;
using namespace cv;

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );


struct cost_function_define
{
  cost_function_define(Point3f p1,Point2f p2):_p1(p1),_p2(p2){}
  template<typename T>
  bool operator()(const T* const cere_r,const T* const cere_t,T* residual)const
  {
    T p_1[3];
    T p_2[3];
    p_1[0]=T(_p1.x);
    p_1[1]=T(_p1.y);
    p_1[2]=T(_p1.z);
cout<<"point_3d: "<<p_1[0]<<" "<<p_1[1]<<"  "<<p_1[2]<<endl;
    AngleAxisRotatePoint(cere_r,p_1,p_2);
    //先旋转再平移
    p_2[0]=p_2[0]+cere_t[0];
    p_2[1]=p_2[1]+cere_t[1];
    p_2[2]=p_2[2]+cere_t[2];

    const T x=p_2[0]/p_2[2];
    const T y=p_2[1]/p_2[2];
    //三维点重投影计算的像素坐标
    const T u=x*520.9+325.1;
    const T v=y*521.0+249.7;
    
   
    //观测的在图像坐标下的值
    T u1=T(_p2.x);
    T v1=T(_p2.y);
 
    residual[0]=u-u1;
    residual[1]=v-v1;
    return true;
  }
   Point3f _p1;
   Point2f _p2;
};


int main ( int argc, char** argv )
{
    if ( argc != 5 )
    {
        cout<<"usage: pose_estimation_3d2d img1 img2 depth1 depth2"<<endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    // 建立3D点
    Mat d1 = imread ( argv[3], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for ( DMatch m:matches )
    {
        ushort d = d1.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/1000.0;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        pts_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
        pts_2d.push_back ( keypoints_2[m.trainIdx].pt );
    }

    cout<<"3d-2d pairs: "<<pts_3d.size() <<endl;

    Mat r, t;
    solvePnP ( pts_3d, pts_2d, K, Mat(), r, t, false ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    Mat R;
    cv::Rodrigues ( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    cout<<"----------------optional berore--------------------"<<endl;
    cout<<"R="<<endl<<R<<endl;
    cout<<"t="<<endl<<t<<endl;

    cout<<"R_inv = "<<R.t() <<endl;
    cout<<"t_inv = "<<-R.t() *t<<endl;

   
    cout<<"calling bundle adjustment"<<endl;

 //给rot，和tranf初值
     double cere_rot[3],cere_tranf[3];
    //  cere_rot[0]=r.at<double>(0,0);
    //  cere_rot[1]=r.at<double>(1,0);
    //  cere_rot[2]=r.at<double>(2,0);
      cere_rot[0]=0;
      cere_rot[1]=1;
      cere_rot[2]=2;

     cere_tranf[0]=t.at<double>(0,0);
     cere_tranf[1]=t.at<double>(1,0);
     cere_tranf[2]=t.at<double>(2,0);


 ceres::Problem problem;
  for(int i=0;i<pts_3d.size();i++)
  {
    ceres::CostFunction* costfunction=new ceres::AutoDiffCostFunction<cost_function_define,2,3,3>(new cost_function_define(pts_3d[i],pts_2d[i]));
    problem.AddResidualBlock(costfunction,NULL,cere_rot,cere_tranf);//注意，cere_rot不能为Mat类型      
  }
 

  ceres::Solver::Options option;
  option.linear_solver_type=ceres::DENSE_SCHUR;
  //输出迭代信息到屏幕
  option.minimizer_progress_to_stdout=true;
  //显示优化信息
  ceres::Solver::Summary summary;
  //开始求解
  ceres::Solve(option,&problem,&summary);
  //显示优化信息
  cout<<summary.BriefReport()<<endl;

 cout<<"----------------optional after--------------------"<<endl;

Mat cam_3d = ( Mat_<double> ( 3,1 )<<cere_rot[0],cere_rot[1],cere_rot[2]);
Mat cam_9d; //旋转向量到->旋转矩阵
cv::Rodrigues ( cam_3d, cam_9d ); // r为旋转向量形式，用Rodrigues公式转换为矩阵

cout<<"cam_9d:"<<endl<<cam_9d<<endl;

cout<<"cam_t:"<<cere_tranf[0]<<"  "<<cere_tranf[1]<<"  "<<cere_tranf[2]<<endl;



}

void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}