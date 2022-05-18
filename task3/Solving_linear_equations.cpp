#include <iostream>
#include <fstream>
#include <iomanip>

#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/SVD>

using namespace std;
void input(string &filename, vector<double> &x_1, vector<double> &y_1, Eigen::MatrixXd &A, Eigen::VectorXd &b);
int main(int argc, char **argv)
{

    string data1_filename = "/home/docker_file/Nreal_training/data.txt";
    string data2_filename = "/home/docker_file/Nreal_training/data2.txt";

    vector<double> x_1, y_1;
    vector<double> x_2, y_2;

    Eigen::MatrixXd A1, A2;
    Eigen::VectorXd b1, b2;

    input(data1_filename, x_1, y_1, A1, b1);
    input(data2_filename, x_2, y_2, A2, b2);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd1(A1,Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd2(A2,Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::Vector2d x_solve_1_0 = A1.colPivHouseholderQr().solve(b1);
    Eigen::Vector2d x_solve_1_1 = svd1.solve(b1);
    Eigen::Vector2d x_solve_2 = A2.colPivHouseholderQr().solve(b2);

    cout <<"QR分解:" <<x_solve_1_0.transpose() << endl;
    cout <<"解1:"<<x_solve_1_1.transpose() << endl;
    cout <<"解2:"<< x_solve_2.transpose() << endl;

    float condition_num1,condition_num2;
    //奇异值矩阵就是从大到小排列的
    condition_num1=svd1.singularValues()(0) / svd1.singularValues()(svd1.singularValues().size()-1);
    condition_num2=svd2.singularValues()(0) / svd2.singularValues()(svd2.singularValues().size()-1);
    cout<<"A1条件数："<<condition_num1<<"\n"<<"A2条件数："<<condition_num2<<endl;
}

void input(string &filename, vector<double> &x, vector<double> &y, Eigen::MatrixXd &A, Eigen::VectorXd &b)
{

    ifstream data1(filename);

    if (!data1.is_open())
    {
        cout << "Open file failed!" << endl;
        exit(EXIT_FAILURE); //若打开失败则中断程序
    }
    else
        cout << "Open file sucess!" << endl;

    string str_txt;
    getline(data1, str_txt);

    while (!data1.eof()) //判断读取是否到达文件末尾
    {
        string str_txt;
        getline(data1, str_txt); //将一整行数据作为一个字符串读入
        if (!str_txt.empty())    //判断该行数据是否读入成功
        {
            //当文本的某一行既包含字符串又包含数字,或者包含多个类别的文本元素时，
            //如果想一次读取一行文本，并对该行的每一个元素分别处理可以使用stringstream类
            stringstream ss;
            ss << str_txt;
            double num1, num2;
            ss >>num1 >> num2;
            x.push_back(num1);
            y.push_back(num2);
        }
    }

    A.resize(x.size(), 2);
    b.resize(y.size());

    for (int i = 0; i < x.size(); i++)
    {
        A(i, 0) = x[i];
        A(i, 1) = 1;
    }

    for (int i = 0; i < y.size(); i++)
    {
        b[i] = y[i];
    }
}
