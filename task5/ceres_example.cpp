#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <spdlog/spdlog.h>
#include <ceres/ceres.h>
using namespace std;
using namespace cv;

class MotionSim
{
public:
    static void TestMe(int argc, char** argv)
    {
        MotionSim motionSim(false);
        motionSim.camFovX = 45;
        motionSim.camFovY = 30;
        motionSim.camRand = 10;
        motionSim.enableVerbose = false;
        motionSim.runMotion(false, false, 7);
        motionSim.visMotion();
    }

public:
    struct MotionView
    {
        Mat_<double> r = Mat_<double>(3, 1);
        Mat_<double> t = Mat_<double>(3, 1);
        Mat_<double> q = Mat_<double>(4, 1);
        Mat_<double> rt = Mat_<double>(6, 1);
        Mat_<double> radian = Mat_<double>(3, 1);
        Mat_<double> degree = Mat_<double>(3, 1);
        Mat_<double> R = Mat_<double>(3, 3);
        Mat_<double> T = Mat_<double>(3, 4);
        Mat_<double> K;
        Mat_<double> D;
        Mat_<Vec3d> point3D;
        Mat_<Vec2d> point2D;
        Mat_<int> point3DIds;
        string print(string savePath = "")
        {
            string str;
            str += fmt::format("r: {}\n", cvarr2str(r.t()));
            str += fmt::format("t: {}\n", cvarr2str(t.t()));
            str += fmt::format("q: {}\n", cvarr2str(q.t()));
            str += fmt::format("rt: {}\n", cvarr2str(rt.t()));
            str += fmt::format("radian: {}\n", cvarr2str(radian.t()));
            str += fmt::format("degree: {}\n", cvarr2str(degree.t()));
            str += fmt::format("R: {}\n", cvarr2str(R));
            str += fmt::format("T: {}\n", cvarr2str(T));
            str += fmt::format("K: {}\n", cvarr2str(K));
            str += fmt::format("D: {}\n", cvarr2str(D.t()));
            if (savePath.empty() == false) { FILE* out = fopen(savePath.c_str(), "w"); fprintf(out, str.c_str()); fclose(out); }
            return str;
        }
    };
    static string cvarr2str(InputArray v)
    {
        Ptr<Formatted> fmtd = cv::format(v, Formatter::FMT_DEFAULT);
        string dst; fmtd->reset();
        for (const char* str = fmtd->next(); str; str = fmtd->next()) dst += string(str);
        return dst;
    }
    static void euler2matrix(double e[3], double R[9], bool forward = true, int argc = 0, char** argv = 0)
    {
        if (argc > 0)
        {
            int N = 999;
            for (int k = 0; k < N; ++k)//OpenCV not better than DIY
            {
                //1.GenerateData
                Matx31d radian0 = radian0.randu(-3.14159265358979323846, 3.14159265358979323846);
                Matx33d R; euler2matrix(radian0.val, R.val, true);
                const double deg2rad = 3.14159265358979323846 * 0.0055555555555555556;
                const double rad2deg = 180 * 0.3183098861837906715;

                //2.CalcByOpenCV
                Matx31d radian1 = cv::RQDecomp3x3(R, Matx33d(), Matx33d()) * deg2rad;

                //3.CalcByDIY
                Matx31d radian2; euler2matrix(R.val, radian2.val, false);

                //4.AnalyzeError
                double infRadian0Radian1 = norm(radian0, radian1, NORM_INF);
                double infRadian1Radian2 = norm(radian1, radian2, NORM_INF);

                //5.PrintError
                cout << endl << "LoopCount: " << k << endl;
                if (infRadian0Radian1 > 0 || infRadian1Radian2 > 0)
                {
                    cout << endl << "5.1PrintError" << endl;
                    cout << endl << "infRadian0Radian1: " << infRadian0Radian1 << endl;
                    cout << endl << "infRadian1Radian2: " << infRadian1Radian2 << endl;
                    if (0)
                    {
                        cout << endl << "5.2PrintDiff" << endl;
                        cout << endl << "radian0-degree0:" << endl << radian0.t() << endl << radian0.t() * rad2deg << endl;
                        cout << endl << "radian1-degree1:" << endl << radian1.t() << endl << radian1.t() * rad2deg << endl;
                        cout << endl << "radian2-degree2:" << endl << radian2.t() << endl << radian2.t() * rad2deg << endl;
                        cout << endl << "5.3PrintOthers" << endl;
                        cout << endl << "R:" << endl << R << endl;
                    }
                    cout << endl << "Press any key to continue" << endl; std::getchar();
                }
            }
            return;
        }
        if (forward)//check with 3D Rotation Converter
        {
            double sinR = std::sin(e[0]);
            double sinP = std::sin(e[1]);
            double sinY = std::sin(e[2]);
            double cosR = std::cos(e[0]);
            double cosP = std::cos(e[1]);
            double cosY = std::cos(e[2]);

            //RPY indicates: first Yaw aroundZ, second Pitch aroundY, third Roll aroundX
            R[0] = cosY * cosP; R[1] = cosY * sinP * sinR - sinY * cosR; R[2] = cosY * sinP * cosR + sinY * sinR;
            R[3] = sinY * cosP; R[4] = sinY * sinP * sinR + cosY * cosR; R[5] = sinY * sinP * cosR - cosY * sinR;
            R[6] = -sinP;       R[7] = cosP * sinR;                      R[8] = cosP * cosR;
        }
        else
        {
            double vs1 = std::abs(R[6] - 1.);
            double vs_1 = std::abs(R[6] + 1.);
            if (vs1 > 1E-9 && vs_1 > 1E-9)
            {
                e[2] = std::atan2(R[3], R[0]); //Yaw aroundZ
                e[1] = std::asin(-R[6]);//Pitch aroundY
                e[0] = std::atan2(R[7], R[8]); //Roll aroundX
            }
            else if (vs_1 <= 1E-9)
            {
                e[2] = 0; //Yaw aroundZ
                e[1] = 3.14159265358979323846 * 0.5;//Pitch aroundY
                e[0] = e[2] + atan2(R[1], R[2]); //Roll aroundX
            }
            else
            {
                e[2] = 0; //Yaw aroundZ
                e[1] = -3.14159265358979323846 * 0.5;//Pitch aroundY
                e[0] = -e[2] + atan2(-R[1], -R[2]); //Roll aroundX
            }
        }
    };
    static void quat2matrix(double q[4], double R[9], bool forward = true)
    {
        if (forward)//refer to qglviwer
        {
            double L1 = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            if (std::abs(L1 - 1) > 1E-9) { std::printf("Not uint quaternion: NormQ=%.9f\n", L1); abort(); }

            double xx = 2.0 * q[1] * q[1];
            double yy = 2.0 * q[2] * q[2];
            double zz = 2.0 * q[3] * q[3];

            double xy = 2.0 * q[1] * q[2];
            double xz = 2.0 * q[1] * q[3];
            double wx = 2.0 * q[1] * q[0];

            double yz = 2.0 * q[2] * q[3];
            double wy = 2.0 * q[2] * q[0];

            double wz = 2.0 * q[3] * q[0];

            R[0] = 1.0 - yy - zz;
            R[4] = 1.0 - xx - zz;
            R[8] = 1.0 - xx - yy;

            R[1] = xy - wz;
            R[3] = xy + wz;

            R[2] = xz + wy;
            R[6] = xz - wy;

            R[5] = yz - wx;
            R[7] = yz + wx;
        }
        else
        {
            double onePlusTrace = 1.0 + R[0] + R[4] + R[8];// Compute one plus the trace of the matrix
            if (onePlusTrace > 1E-9)
            {
                double s = sqrt(onePlusTrace) * 2.0;
                double is = 1 / s;
                q[0] = 0.25 * s;
                q[1] = (R[7] - R[5]) * is;
                q[2] = (R[2] - R[6]) * is;
                q[3] = (R[3] - R[1]) * is;
            }
            else
            {
                std::printf("1+trace(R)=%.9f is too small and (R11,R22,R33)=(%.9f,%.9f,%.9f)\n", onePlusTrace, R[0], R[4], R[8]);
                if ((R[0] > R[4]) && (R[0] > R[8]))//max(R00, R11, R22)=R00
                {
                    double s = sqrt(1.0 + R[0] - R[4] - R[8]) * 2.0;
                    double is = 1 / s;
                    q[0] = (R[5] - R[7]) * is;
                    q[1] = 0.25 * s;
                    q[2] = (R[1] + R[3]) * is;
                    q[3] = (R[2] + R[6]) * is;
                }
                else if (R[4] > R[8])//max(R00, R11, R22)=R11
                {
                    double s = sqrt(1.0 - R[0] + R[4] - R[8]) * 2.0;
                    double is = 1 / s;
                    q[0] = (R[2] - R[6]) * is;
                    q[1] = (R[1] + R[3]) * is;
                    q[2] = 0.25 * s;
                    q[3] = (R[5] + R[7]) * is;
                }
                else//max(R00, R11, R22)=R22
                {
                    double s = sqrt(1.0 - R[0] - R[4] + R[8]) * 2.0;
                    double is = 1 / s;
                    q[0] = (R[1] - R[3]) * is;
                    q[1] = (R[2] + R[6]) * is;
                    q[2] = (R[5] + R[7]) * is;
                    q[3] = 0.25 * s;
                }
            }
            double L1 = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            if (L1 < 1e-9) { std::printf("Wrong rotation matrix: NormQ=%.9f\n", L1); abort(); }
            else { L1 = 1 / L1; q[0] *= L1; q[1] *= L1; q[2] *= L1; q[3] *= L1; }
        }
    }
    static void vec2quat(double r[3], double q[4], bool forward = true)
    {
        if (forward)//refer to qglviwer
        {
            double theta = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            if (std::abs(theta) < 1E-9)
            {
                q[0] = 1; q[1] = q[2] = q[3] = 0;
                std::printf("Rotation approximates zero: Theta=%.9f\n", theta);
            };

            q[0] = std::cos(theta * 0.5);
            double ss = std::sin(theta * 0.5) / theta;
            q[1] = r[0] * ss;
            q[2] = r[1] * ss;
            q[3] = r[2] * ss;
        }
        else
        {
            double L1 = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            if (std::abs(L1 - 1) > 1E-9) { std::printf("Not uint quaternion: NormQ=%.9f\n", L1); abort(); }

            double theta = 2 * acos(q[0]);
            if (theta > 3.14159265358979323846) theta = 2 * 3.14159265358979323846 - theta;
            double thetaEx = theta / std::sin(theta * 0.5);
            r[0] = q[1] * thetaEx;
            r[1] = q[2] * thetaEx;
            r[2] = q[3] * thetaEx;
        }
    }
    static void vec2matrix(double r[3], double R[9], bool forward = true, int argc = 0, char** argv = 0)
    {
        if (argc > 0)
        {
            int N = 999;
            for (int k = 0; k < N; ++k) //refer to the subsequent article for more details
            {
                //1.GenerateData
                Matx31d r0 = r0.randu(-999, 999);
                Matx33d R0; cv::Rodrigues(r0, R0);

                //2.CalcByOpenCV
                Matx33d R1;
                Matx31d r1;
                cv::Rodrigues(r0, R1);
                cv::Rodrigues(R0, r1);

                //3.CalcByDIY
                Matx33d R2;
                Matx31d r2;
                vec2matrix(r0.val, R2.val, true);
                vec2matrix(r2.val, R0.val, false);

                //4.AnalyzeError
                double infR1R2 = norm(R1, R2, NORM_INF);
                double infr1r2 = norm(r1, r2, NORM_INF);

                //5.PrintError
                cout << endl << "LoopCount: " << k << endl;
                if (infR1R2 > 1E-12 || infr1r2 > 1E-12)
                {
                    cout << endl << "5.1PrintError" << endl;
                    cout << endl << "infR1R2: " << infR1R2 << endl;
                    cout << endl << "infr1r2: " << infr1r2 << endl;
                    if (0)
                    {
                        cout << endl << "5.2PrintDiff" << endl;
                        cout << endl << "R1: " << endl << R1 << endl;
                        cout << endl << "R2: " << endl << R2 << endl;
                        cout << endl;
                        cout << endl << "r1: " << endl << r1.t() << endl;
                        cout << endl << "r2: " << endl << r2.t() << endl;
                        cout << endl << "5.3PrintOthers" << endl;
                    }
                    cout << endl << "Press any key to continue" << endl; std::getchar();
                }
            }
            return;
        }

        if (forward)
        {
            double theta = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            if (theta < 1E-9)
            {
                R[0] = R[4] = R[8] = 1.0;
                R[1] = R[2] = R[3] = R[5] = R[6] = R[7] = 0.0;
                std::printf("Rotation approximates zero: Theta=%.9f\n", theta);
                return;
            }
            double cs = cos(theta);
            double sn = sin(theta);
            double itheta = 1. / theta;
            double cs1 = 1 - cs;
            double nx = r[0] * itheta;
            double ny = r[1] * itheta;
            double nz = r[2] * itheta;

            double nxnx = nx * nx, nyny = ny * ny, nznz = nz * nz;
            double nxny = nx * ny, nxnz = nx * nz, nynz = ny * nz;
            double nxsn = nx * sn, nysn = ny * sn, nzsn = nz * sn;

            R[0] = nxnx * cs1 + cs;
            R[3] = nxny * cs1 + nzsn;
            R[6] = nxnz * cs1 - nysn;

            R[1] = nxny * cs1 - nzsn;
            R[4] = nyny * cs1 + cs;
            R[7] = nynz * cs1 + nxsn;

            R[2] = nxnz * cs1 + nysn;
            R[5] = nynz * cs1 - nxsn;
            R[8] = nznz * cs1 + cs;

            if (0)
            {
                Mat_<double> dRdu({ 9, 4 }, {
                    2 * nx * cs1, 0, 0, (nxnx - 1) * sn,
                    ny * cs1, nx * cs1, -sn, nxny * sn - nz * cs,
                    nz * cs1, sn, nx * cs1, nxnz * sn + ny * cs,
                    ny * cs1, nx * cs1, sn, nxny * sn + nz * cs,
                    0, 2 * ny * cs1, 0, (nyny - 1) * sn,
                    -sn, nz * cs1, ny * cs1, nynz * sn - nx * cs,
                    nz * cs1, -sn, nx * cs1, nxnz * sn - ny * cs,
                    sn, nz * cs1, ny * cs1, nynz * sn + nx * cs,
                    0, 0, 2 * nz * cs1, (nznz - 1) * sn });

                Mat_<double> dudv({ 4, 4 }, {
                    itheta, 0, 0, -nx * itheta,
                    0, itheta, 0, -ny * itheta,
                    0, 0, itheta, -nz * itheta,
                    0, 0, 0, 1 });

                Mat_<double> dvdr({ 4, 3 }, {
                    1, 0, 0,
                    0, 1, 0,
                    0, 0, 1,
                    nx, ny, nz });

                Mat_<double> Jacobian = dRdu * dudv * dvdr;//rows=9 cols=3
            }
        }
        else
        {
            double sx = R[7] - R[5];
            double sy = R[2] - R[6];
            double sz = R[3] - R[1];
            double sn = sqrt(sx * sx + sy * sy + sz * sz) * 0.5;
            double cs = (R[0] + R[4] + R[8] - 1) * 0.5;
            double theta = acos(cs);
            double ss = 2 * sn;
            double iss = 1. / ss;
            double tss = theta * iss;
            r[0] = tss * sx;
            r[1] = tss * sy;
            r[2] = tss * sz;

            if (0)
            {
                Mat_<double> drdu({ 3, 4 }, {
                    tss, 0, 0, (sn - theta * cs) * iss * iss * sx * 2,
                    0, tss, 0, (sn - theta * cs) * iss * iss * sy * 2,
                    0, 0, tss, (sn - theta * cs) * iss * iss * sz * 2 });

                Mat_<double> dudR({ 4, 9 }, {
                    0, 0, 0, 0, 0, -1, 0, 1, 0,
                    0, 0, 1, 0, 0, 0, -1, 0, 0,
                    0, -1, 0, 1, 0, 0, 0, 0, 0,
                    -iss, 0, 0, 0, -iss, 0, 0, 0, -iss });

                Mat_<double> Jacobian = drdu * dudR;//rows=3 cols=9
            }
        }
    }

private:
    const int nHorPoint3D = 100;
    const int nVerPoint3D = 100;
    const double varPoint3DXY = 10.;
    const double minPoint3DZ = 1.;
    const double maxPoint3DZ = 99.;
    const double minCamZ = 101.;
    const double maxCamZ = 150.;
    const double varCamDegree = 10.;
    Mat_<Vec3d> allPoint3D = Mat_<Vec3d>(nVerPoint3D * nHorPoint3D, 1);
    Mat_<double> allPoint3DZ = Mat_<double>(nVerPoint3D * nHorPoint3D, 1);
    Mat_<double> K;
    Mat_<double> D;
    const double deg2rad = 3.14159265358979323846 * 0.0055555555555555556;
    const double rad2deg = 180 * 0.3183098861837906715;

public:
    int camRows = 480;
    int camCols = 640;
    int camFovY = 90;
    int camFovX = 90;
    int camRand = 10;//append random[0,camRand] to camera intrinsics
    int nCamDist = 5;//refer to opencv for value domain
    int nMinMotion = 32; // no less than X motion views
    int nMaxMotion = INT_MAX; // no more than X motion views
    int nPoint2DThenExit = 32;//exit when less than X pixies
    int rotMode = 1 + 2 + 4;//0=noRot 1=xAxis 2=yAxis 4=zAxis
    bool noTrans = false;//translate or not while motion
    bool world2D = false;//planar world or not
    bool rndSeek = true;//use random seek or not
    bool enableVerbose = false;//check motions one by one or not
    vector<MotionView> motionViews;//World Information: RightX, FrontY, DownZ
    MotionSim(bool run = true, bool world2D0 = false, bool noTrans0 = false, int rotMode0 = 7) { if (run) runMotion(world2D0, noTrans0, rotMode0); }

public:
    void runMotion(bool world2D0 = false, bool noTrans0 = false, int rotMode0 = 7)
    {
        world2D = world2D0;
        noTrans = noTrans0;
        rotMode = rotMode0;
        motionViews.clear();
        if (rndSeek) cv::setRNGSeed(clock());
        while (motionViews.size() < nMinMotion)
        {
            //1.GetAllPoint3D
            if (world2D) allPoint3DZ = 0.;
            else cv::randu(allPoint3DZ, -maxPoint3DZ, -minPoint3DZ);//DownZ
            for (int i = 0, k = 0; i < nVerPoint3D; ++i)
                for (int j = 0; j < nHorPoint3D; ++j, ++k)
                    allPoint3D(k) = Vec3d((j + cv::randu<double>()) * varPoint3DXY, (i + cv::randu<double>()) * varPoint3DXY, allPoint3DZ(i, j));

            //2.GetCamParams
            double camFx = camCols / 2. / std::tan(camFovX / 2. * deg2rad) + cv::randu<double>() * camRand;
            double camFy = camRows / 2. / std::tan(camFovY / 2. * deg2rad) + cv::randu<double>() * camRand;
            double camCx = camCols / 2. + cv::randu<double>() * camRand;
            double camCy = camRows / 2. + cv::randu<double>() * camRand;
            K.create(3, 3); K << camFx, 0, camCx, 0, camFy, camCy, 0, 0, 1;
            D.create(nCamDist, 1); cv::randu(D, -1.0, 1.0);

            //3.GetAllMotionView
            motionViews.clear();
            for (int64 k = 0; ; ++k)
            {
                //3.1 JoinCamParams
                MotionView view;
                view.K = K.clone();
                view.D = D.clone();

                //3.2 GetCamTrans
                if (k == 0) view.t(0) = view.t(1) = 0;
                else
                {
                    view.t(0) = motionViews[k - 1].t(0) + cv::randu<double>() * varPoint3DXY;
                    view.t(1) = motionViews[k - 1].t(1) + cv::randu<double>() * varPoint3DXY;
                }
                view.t(2) = minCamZ + cv::randu<double>() * (maxCamZ - minCamZ);
                view.t(2) = -view.t(2);//DownZ
                if (noTrans && k != 0) { view.t(0) = motionViews[0].t(0); view.t(1) = motionViews[0].t(1); view.t(2) = motionViews[0].t(2); }

                //3.3 GetCamRot: degree-->radian-->matrix-->vector&quaternion
                view.degree = 0.;
                if (rotMode & 1) view.degree(0) = cv::randu<double>() * varCamDegree;
                if (rotMode & 2) view.degree(1) = cv::randu<double>() * varCamDegree;
                if (rotMode & 4) view.degree(2) = cv::randu<double>() * varCamDegree;
                view.radian = view.degree * deg2rad;
                euler2matrix(view.radian.ptr<double>(), view.R.ptr<double>());
                cv::Rodrigues(view.R, view.r);
                quat2matrix(view.q.ptr<double>(), view.R.ptr<double>(), false);
                cv::hconcat(view.R, view.t, view.T);
                cv::vconcat(view.r, view.t, view.rt);

                //3.4 GetPoint3DAndPoint2D
                Mat_<Vec2d> allPoint2D;
                cv::projectPoints(allPoint3D, -view.r, -view.R.t() * view.t, view.K, view.D, allPoint2D);
                for (int k = 0; k < allPoint2D.total(); ++k)
                    if (allPoint2D(k)[0] > 0 && allPoint2D(k)[0] < camCols && allPoint2D(k)[1] > 0 && allPoint2D(k)[1] < camRows)
                    {
                        view.point2D.push_back(allPoint2D(k));
                        view.point3D.push_back(allPoint3D(k));
                        view.point3DIds.push_back(k);
                    }

                //3.5 PrintDetails
                motionViews.push_back(view);
                if (enableVerbose)
                {
                    cout << endl << view.print();
                    cout << fmt::format("view={}   features={}\n", k, view.point2D.rows);
                    double minV = 0, maxV = 0;//Distortion makes some minV next to maxV
                    int minId = 0, maxId = 0;
                    cv::minMaxIdx(allPoint2D.reshape(1, int(allPoint2D.total()) * allPoint2D.channels()), &minV, &maxV, &minId, &maxId);
                    cout << fmt::format("minInfo:({}, {})", minId, minV) << allPoint3D(minId / 2) << allPoint2D(minId / 2) << endl;
                    cout << fmt::format("maxInfo:({}, {})", maxId, maxV) << allPoint3D(maxId / 2) << allPoint2D(maxId / 2) << endl;
                    cout << "Press any key to continue" << endl; std::getchar();
                }
                if (view.point2D.rows < nPoint2DThenExit || motionViews.size() > nMaxMotion) break;
            }
        }
    }
    void visMotion()
    {
        //1.CreateWidgets
        Size2d validSize(nHorPoint3D * varPoint3DXY, nVerPoint3D * varPoint3DXY);
        Mat_<cv::Affine3d> camPoses(int(motionViews.size()), 1); for (int k = 0; k < camPoses.rows; ++k) camPoses(k) = cv::Affine3d(motionViews[k].T);
        viz::WText worldInfo(fmt::format("nMotionView: {}\nK: {}\nD: {}", motionViews.size(), cvarr2str(K), cvarr2str(D)), Point(10, 240), 10);
        viz::WCoordinateSystem worldCSys(1000);
        viz::WPlane worldGround(Point3d(validSize.width / 2, validSize.height / 2, 0), Vec3d(0, 0, 1), Vec3d(0, 1, 0), validSize);
        viz::WCloud worldPoints(allPoint3D, Mat_<Vec3b>(allPoint3D.size(), Vec3b(0, 255, 0)));
        viz::WTrajectory camTraj1(camPoses, viz::WTrajectory::FRAMES, 8);
        viz::WTrajectorySpheres camTraj2(camPoses, 100, 2);
        viz::WTrajectoryFrustums camTraj3(camPoses, Matx33d(K), 4., viz::Color::yellow());
        worldCSys.setRenderingProperty(viz::OPACITY, 0.1);
        worldGround.setRenderingProperty(viz::OPACITY, 0.1);
        camTraj2.setRenderingProperty(viz::OPACITY, 0.6);

        //2.ShowWidgets
        static viz::Viz3d viz3d(__FUNCTION__);
        viz3d.showWidget("worldInfo", worldInfo);
        viz3d.showWidget("worldCSys", worldCSys);
        viz3d.showWidget("worldGround", worldGround);
        viz3d.showWidget("worldPoints", worldPoints);
        viz3d.showWidget("camTraj1", camTraj1);
        viz3d.showWidget("camTraj2", camTraj2);
        viz3d.showWidget("camTraj3", camTraj3);

        //3.UpdateWidghts
        static const vector<MotionView>& views = motionViews;
        viz3d.registerKeyboardCallback([](const viz::KeyboardEvent& keyboarEvent, void* pVizBorad)->void
            {
                if (keyboarEvent.action != viz::KeyboardEvent::KEY_DOWN) return;
                static int pos = 0;
                if (keyboarEvent.code == ' ')
                {
                    size_t num = views.size();
                    size_t ind = pos % num;
                    double xmin3D = DBL_MAX, ymin3D = DBL_MAX, xmin2D = DBL_MAX, ymin2D = DBL_MAX;
                    double xmax3D = -DBL_MAX, ymax3D = -DBL_MAX, xmax2D = -DBL_MAX, ymax2D = -DBL_MAX;
                    for (size_t k = 0; k < views[ind].point3D.rows; ++k)
                    {
                        Vec3d pt3 = views[ind].point3D(int(k));
                        Vec2d pt2 = views[ind].point2D(int(k));
                        if (pt3[0] < xmin3D) xmin3D = pt3[0];
                        if (pt3[0] > xmax3D) xmax3D = pt3[0];
                        if (pt3[1] < ymin3D) ymin3D = pt3[1];
                        if (pt3[1] > ymax3D) ymax3D = pt3[1];
                        if (pt2[0] < xmin2D) xmin2D = pt2[0];
                        if (pt2[0] > xmax2D) xmax2D = pt2[0];
                        if (pt2[1] < ymin2D) ymin2D = pt2[1];
                        if (pt2[1] > ymax2D) ymax2D = pt2[1];
                    }
                    if (pos != 0)
                    {
                        for (int k = 0; k < views[ind == 0 ? num - 1 : ind - 1].point3D.rows; ++k) viz3d.removeWidget("active" + std::to_string(k));
                        viz3d.removeWidget("viewInfo");
                        viz3d.removeWidget("camSolid");
                    }
                    for (int k = 0; k < views[ind].point3D.rows; ++k) viz3d.showWidget("active" + std::to_string(k), viz::WSphere(views[ind].point3D(k), 5, 10));
                    viz3d.showWidget("viewInfo", viz::WText(fmt::format("CurrentMotion: {}\nValidPoints: {}\nMin3DXY_Min2DXY: {}, {}, {}, {}\nMax3DXY_Max2DXY: {}, {}, {}, {}\nRot_Trans_Euler: {}\n",
                        ind, views[ind].point3D.rows, xmin3D, ymin3D, xmin2D, ymin2D, xmax3D, ymax3D, xmax2D, ymax2D,
                        cvarr2str(views[ind].r.t()) + cvarr2str(views[ind].t.t()) + cvarr2str(views[ind].degree.t())), Point(10, 10), 10));
                    viz3d.showWidget("camSolid", viz::WCameraPosition(Matx33d(views[ind].K), 10, viz::Color::yellow()), cv::Affine3d(views[ind].T));
                    ++pos;
                }
            }, 0);
        viz3d.spin();
    }
};

class OptimizeRt
{
public:
    using MotionView = MotionSim::MotionView;
    static void TestMe(int argc = 0, char** argv = 0)
    {
        int N = 99;
        for (int k = 0; k < N; ++k)
        {
            //1.GenerateData
            bool world2D = k % 2;
            int rotMode = k % 7 + 1;
            MotionSim motionSim(false);
            motionSim.camFovX = 90;
            motionSim.camFovY = 90;
            motionSim.camRand = 10;
            motionSim.nMinMotion = 16;//2
            motionSim.nMaxMotion = 32;//4
            motionSim.rndSeek = false;
            motionSim.nCamDist = 5;
            motionSim.runMotion(world2D, false, rotMode);
            //motionSim.visMotion();
            int rndInd = int(motionSim.motionViews.size() * cv::randu<double>());
            Mat_<double> r0 = -motionSim.motionViews[rndInd].r;
            Mat_<double> t0 = -motionSim.motionViews[rndInd].R.t() * motionSim.motionViews[rndInd].t;
            const MotionView& motionView = motionSim.motionViews[rndInd];
            double errRatio = 0.9;

            //2.CalcByCeres
            Mat_<double> param1; cv::vconcat(r0, t0, param1); param1 *= errRatio;//use one group of param for LMSolver param format
            ceres::Problem problem;
            problem.AddResidualBlock(new CeresCostRt(motionView), NULL, param1.ptr<double>());
            ceres::Solver::Options options;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            int nIter1 = (int)summary.iterations.size();

            //3.CalcByOpenCV
            Mat_<double> param2; cv::vconcat(r0, t0, param2); param2 *= errRatio;
            Ptr<cv::LMSolver::Callback> callback = makePtr<CvCallbackRt>(motionView);
            Ptr<cv::LMSolver> lmSolver2 = cv::LMSolver::create(callback, 50);
            int nIter2 = lmSolver2->run(param2);

            //4.AnalyzeError
            double infr0r0 = norm(r0, r0 * errRatio, NORM_INF);
            double infr0r1 = norm(r0, param1.rowRange(0, 3), NORM_INF);
            double infr0r2 = norm(r0, param2.rowRange(0, 3), NORM_INF);
            double inft0t0 = norm(t0, t0 * errRatio, NORM_INF);
            double inft0t1 = norm(t0, param1.rowRange(3, 6), NORM_INF);
            double inft0t2 = norm(t0, param2.rowRange(3, 6), NORM_INF);
            double infr1r2 = norm(param1.rowRange(0, 3), param2.rowRange(0, 3), NORM_INF);
            double inft1t2 = norm(param1.rowRange(3, 6), param2.rowRange(3, 6), NORM_INF);

            //5.PrintError
            cout << fmt::format("LoopCount: {}      CeresSolver.iters: {}      CVLMSolver.iters: {}\n", k, nIter1, nIter2);
            if (infr0r1 > 1e-8 || infr0r2 > 1e-8 || inft0t1 > 1e-8 || inft0t2 > 1e-8)
            {
                cout << fmt::format("infr0r1: {:<15.9}\t\t{:<15.9}\n", infr0r1, infr0r0);
                cout << fmt::format("infr0r2: {:<15.9}\t\t{:<15.9}\n", infr0r2, infr0r0);
                cout << fmt::format("inft0t1: {:<15.9}\t\t{:<15.9}\n", inft0t1, inft0t0);
                cout << fmt::format("inft0t2: {:<15.9}\t\t{:<15.9}\n", inft0t2, inft0t0);
                cout << fmt::format("infr1r2: {:<15.9}\t\t\n", infr1r2);
                cout << fmt::format("inft1t2: {:<15.9}\t\t\n", inft1t2);
                cout << "Press any key to continue" << endl; std::getchar();
            }
        }
    }

public:
    struct CeresCostRt : public ceres::CostFunction
    {
        const MotionView& motionView;//use K&D&point2D&point3D as groundtruth
        CeresCostRt(const MotionView& motionView0) : motionView(motionView0)
        {
            this->set_num_residuals(motionView.point2D.rows * 2);
            this->mutable_parameter_block_sizes()->push_back(6);//Also can use two groups of params: r and t. Refer to OptimizeKDRt. And two groups can contribute to less jacobians computation.
        }
        bool Evaluate(double const* const* params, double* residuals, double** jacobians) const
        {
            //1.ExtractInput
            Vec3d r(params[0]);
            Vec3d t(params[0] + 3);
            Mat_<Vec2d> errPoint2D(motionView.point2D.rows, 1, (Vec2d*)(residuals));
            Mat_<double> dpdrt; if (jacobians && jacobians[0]) dpdrt = Mat_<double>(motionView.point2D.rows * 2, 6, jacobians[0]);

            //2.CalcJacAndErr
            Mat_<double> dpdKDT;
            Mat_<Vec2d> point2DEva;
            cv::projectPoints(motionView.point3D, r, t, motionView.K, motionView.D, point2DEva, dpdrt.empty() ? noArray() : dpdKDT);
            errPoint2D = point2DEva - motionView.point2D;
            if (dpdrt.empty() == false) dpdKDT.colRange(0, 6).copyTo(dpdrt);
            return true;
        }
    };

public:
    struct CvCallbackRt : public cv::LMSolver::Callback
    {
        const MotionView& motionView;//use K&D&point2D&point3D as groundtruth
        CvCallbackRt(const MotionView& motionView0) : motionView(motionView0) {}
        bool compute(InputArray params, OutputArray residuals, OutputArray jacobians) const
        {
            //1.ExtractInput
            Vec3d r(params.getMat().ptr<double>());
            Vec3d t(params.getMat().ptr<double>() + 3);
            if (residuals.empty()) residuals.create(motionView.point2D.rows * 2, 1, CV_64F);
            if (jacobians.needed() && jacobians.empty()) jacobians.create(motionView.point2D.rows * 2, params.rows(), CV_64F);
            Mat_<Vec2d> errPoint2D = residuals.getMat();
            Mat_<double> dpdrt = jacobians.getMat();

            //2.CalcJacAndErr
            Mat_<double> dpdKDT;
            Mat_<Vec2d> point2DEva;
            cv::projectPoints(motionView.point3D, r, t, motionView.K, motionView.D, point2DEva, dpdrt.empty() ? noArray() : dpdKDT);
            errPoint2D = point2DEva - motionView.point2D;
            if (dpdrt.empty() == false) dpdKDT.colRange(0, 6).copyTo(dpdrt);
            return true;
        }
    };
};

class OptimizeKDRt
{
public:
    using MotionView = MotionSim::MotionView;
    static void TestMe(int argc = 0, char** argv = 0)
    {
        int N = 99;
        for (int k = 0; k < N; ++k)
        {
            //1.GenerateData
            bool world2D = k % 2;
            int rotMode = k % 7 + 1;
            MotionSim motionSim(false);
            motionSim.camFovX = 90;
            motionSim.camFovY = 90;
            motionSim.camRand = 10;
            motionSim.nMinMotion = 16;//2
            motionSim.nMaxMotion = 32;//4
            motionSim.rndSeek = false;
            motionSim.nCamDist = 5;
            motionSim.runMotion(world2D, false, rotMode);
            //motionSim.visMotion();
            Mat_<double> rs0; for (size_t k = 0; k < motionSim.motionViews.size(); ++k) rs0.push_back(-motionSim.motionViews[k].r);
            Mat_<double> ts0; for (size_t k = 0; k < motionSim.motionViews.size(); ++k) ts0.push_back(-motionSim.motionViews[k].R.t() * motionSim.motionViews[k].t);
            Mat_<double> K0({ 4, 1 }, { motionSim.motionViews[0].K(0, 0), motionSim.motionViews[0].K(1, 1), motionSim.motionViews[0].K(0, 2), motionSim.motionViews[0].K(1, 2) });
            Mat_<double> D0 = motionSim.motionViews[0].D.clone();
            double errRatio = 0.9;
            double errRatioTrans = 0.99;

            //2.CalcByCeres
            Mat_<double> params1, rs1, ts1;//use one group of param for LMSolver param format
            for (int k = 0; k < rs0.rows; k += 3) { params1.push_back(rs0.rowRange(k, k + 3) * errRatio); params1.push_back(ts0.rowRange(k, k + 3) * errRatioTrans); }
            params1.push_back(K0 * errRatio);
            params1.push_back(D0 * errRatio);
            ceres::Problem problem;
            for (int k = 0; k < motionSim.motionViews.size(); ++k)
                problem.AddResidualBlock(new CeresCostKDRt(motionSim.motionViews[k]), NULL, params1.ptr<double>(k * 6), params1.ptr<double>(k * 6 + 3),
                    params1.ptr<double>(params1.rows - 4 - D0.rows), params1.ptr<double>(params1.rows - D0.rows));
            ceres::Solver::Options options;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            int nIter1 = (int)summary.iterations.size();
            for (int k = 0; k < params1.rows - 4 - D0.rows; k += 6) { rs1.push_back(params1.rowRange(k, k + 3)); ts1.push_back(params1.rowRange(k + 3, k + 6)); }
            Mat_<double> K1 = params1.rowRange(params1.rows - 4 - D0.rows, params1.rows - D0.rows).clone();
            Mat_<double> D1 = params1.rowRange(params1.rows - D0.rows, params1.rows).clone();

            //3.CalcByOpenCV
            Mat_<double> params2, rs2, ts2;
            for (int k = 0; k < rs0.rows; k += 3) { params2.push_back(rs0.rowRange(k, k + 3) * errRatio); params2.push_back(ts0.rowRange(k, k + 3) * errRatioTrans); }
            params2.push_back(K0 * errRatio);
            params2.push_back(D0 * errRatio);
            Ptr<cv::LMSolver::Callback> callback = makePtr<CvCallbackKDRt>(motionSim.motionViews);
            Ptr<cv::LMSolver> lmSolver2 = cv::LMSolver::create(callback, 50);
            int nIter2 = lmSolver2->run(params2);
            for (int k = 0; k < params2.rows - 4 - D0.rows; k += 6) { rs2.push_back(params2.rowRange(k, k + 3)); ts2.push_back(params2.rowRange(k + 3, k + 6)); }
            Mat_<double> K2 = params2.rowRange(params2.rows - 4 - D0.rows, params2.rows - D0.rows).clone();
            Mat_<double> D2 = params2.rowRange(params2.rows - D0.rows, params2.rows).clone();

            //4.AnalyzeError
            double infrs0rs0 = norm(rs0, rs0 * errRatio, NORM_INF);
            double infrs0rs1 = norm(rs0, rs1, NORM_INF);
            double infrs0rs2 = norm(rs0, rs2, NORM_INF);
            double infts0ts0 = norm(ts0, ts0 * errRatioTrans, NORM_INF);
            double infts0ts1 = norm(ts0, ts1, NORM_INF);
            double infts0ts2 = norm(ts0, ts2, NORM_INF);
            double infK0K0 = norm(K0, K0 * errRatio, NORM_INF);
            double infK0K1 = norm(K0, K1, NORM_INF);
            double infK0K2 = norm(K0, K2, NORM_INF);
            double infD0D0 = norm(D0, D0 * errRatio, NORM_INF);
            double infD0D1 = norm(D0, D1, NORM_INF);
            double infD0D2 = norm(D0, D2, NORM_INF);
            double infrs1rs2 = norm(rs1, rs2, NORM_INF);
            double infts1ts2 = norm(ts1, ts2, NORM_INF);
            double infK1K2 = norm(K1, K2, NORM_INF);
            double infD1D2 = norm(D1, D2, NORM_INF);

            //5.PrintError
            cout << fmt::format("LoopCount: {}      CeresSolver.iters: {}      CVLMSolver.iters: {}\n", k, nIter1, nIter2);
            if (infrs0rs1 > 1e-8 || infrs0rs2 > 1e-8 || infts0ts1 > 1e-8 || infts0ts2 > 1e-8 || infK0K1 > 1e-8 || infK0K2 > 1e-8 || infD0D1 > 1e-8 || infD0D2 > 1e-8)
            {
                cout << fmt::format("infrs0rs1: {:<15.9}\t\t{:<15.9}\n", infrs0rs1, infrs0rs0);
                cout << fmt::format("infrs0rs2: {:<15.9}\t\t{:<15.9}\n", infrs0rs2, infrs0rs0);
                cout << fmt::format("infts0ts1: {:<15.9}\t\t{:<15.9}\n", infts0ts1, infts0ts0);
                cout << fmt::format("infts0ts2: {:<15.9}\t\t{:<15.9}\n", infts0ts2, infts0ts0);
                cout << fmt::format("infK0K1  : {:<15.9}\t\t{:<15.9}\n", infK0K1, infK0K0);
                cout << fmt::format("infK0K2  : {:<15.9}\t\t{:<15.9}\n", infK0K2, infK0K0);
                cout << fmt::format("infD0D1  : {:<15.9}\t\t{:<15.9}\n", infD0D1, infD0D0);
                cout << fmt::format("infD0D2  : {:<15.9}\t\t{:<15.9}\n", infD0D2, infD0D0);
                cout << fmt::format("infrs1rs2: {:<15.9}\t\t\n", infrs1rs2);
                cout << fmt::format("infts1ts2: {:<15.9}\t\t\n", infts1ts2);
                cout << fmt::format("infK1D2  : {:<15.9}\t\t\n", infK1K2);
                cout << fmt::format("infD1D2  : {:<15.9}\t\t\n", infD1D2);
                cout << "Press any key to continue" << endl; std::getchar();
            }
        }
    }

public:
    struct CeresCostKDRt : public ceres::CostFunction
    {
        const MotionView& motionView;//use K&D&point2D&point3D as groundtruth
        CeresCostKDRt(const MotionView& motionView0) : motionView(motionView0)
        {
            this->set_num_residuals(motionView.point2D.rows * 2);
            this->mutable_parameter_block_sizes()->push_back(3);
            this->mutable_parameter_block_sizes()->push_back(3);
            this->mutable_parameter_block_sizes()->push_back(4);
            this->mutable_parameter_block_sizes()->push_back(motionView.D.rows);
        }
        bool Evaluate(double const* const* params, double* residuals, double** jacobians) const
        {
            //1.ExtractInput
            Vec3d r(params[0]);
            Vec3d t(params[1]);
            Matx33d K(params[2][0], 0, params[2][2], 0, params[2][1], params[2][3], 0, 0, 1);
            Mat_<double> D(motionView.D.rows, 1); for (int k = 0; k < D.rows; ++k) D(k) = params[3][k];
            Mat_<Vec2d> errPoint2D(motionView.point2D.rows, 1, (Vec2d*)(residuals));
            Mat_<double> dpdr; if (jacobians && jacobians[0]) dpdr = Mat_<double>(motionView.point2D.rows * 2, 3, jacobians[0]);
            Mat_<double> dpdt; if (jacobians && jacobians[1]) dpdt = Mat_<double>(motionView.point2D.rows * 2, 3, jacobians[1]);
            Mat_<double> dpdK; if (jacobians && jacobians[2]) dpdK = Mat_<double>(motionView.point2D.rows * 2, 4, jacobians[2]);
            Mat_<double> dpdD; if (jacobians && jacobians[3]) dpdD = Mat_<double>(motionView.point2D.rows * 2, motionView.D.rows, jacobians[3]);

            //2.CalcJacAndErr
            Mat_<double> dpdKDT;
            Mat_<Vec2d> point2DEva;
            cv::projectPoints(motionView.point3D, r, t, K, D, point2DEva, dpdr.empty() && dpdt.empty() && dpdK.empty() && dpdD.empty() ? noArray() : dpdKDT);
            errPoint2D = point2DEva - motionView.point2D;
            if (dpdr.empty() == false) dpdKDT.colRange(0, 3).copyTo(dpdr);
            if (dpdt.empty() == false) dpdKDT.colRange(3, 6).copyTo(dpdt);
            if (dpdK.empty() == false) dpdKDT.colRange(6, 10).copyTo(dpdK);
            if (dpdD.empty() == false) dpdKDT.colRange(10, 10 + D.rows).copyTo(dpdD);
            return true;
        }
    };

public:
    struct CvCallbackKDRt : public cv::LMSolver::Callback
    {
        int nResidual = 0;
        const vector<MotionView>& motionViews;//use K&D&point2D&point3D as groundtruth
        CvCallbackKDRt(const vector<MotionView>& motionViews0) : motionViews(motionViews0) { for (size_t i = 0; i < motionViews.size(); ++i) nResidual += motionViews[i].point2D.rows * 2; }
        bool compute(InputArray params, OutputArray residuals, OutputArray jacobians) const
        {
            //1.ExtractInputGlobal
            Mat_<double> cvParams = params.getMat();
            double* rsdata = params.getMat().ptr<double>();
            double* tsdata = params.getMat().ptr<double>() + 3;
            double* KData = params.getMat().ptr<double>() + motionViews.size() * 2 * 3;
            double* DData = params.getMat().ptr<double>() + motionViews.size() * 2 * 3 + 4;
            if (residuals.empty()) residuals.create(nResidual, 1, CV_64F);
            if (jacobians.needed() && jacobians.empty()) jacobians.create(nResidual, params.rows(), CV_64F);
            Mat_<Vec2d> errPoint2Ds = residuals.getMat();
            Mat_<double> dpdKDTs = jacobians.getMat();

            //2.CalcJacAndErrGlobal
            for (int k = 0, row1 = 0, row2 = 0, nView = int(motionViews.size()); k < nView; ++k, row1 = row2)
            {
                const MotionView& motionView = motionViews[k];
                //2.1 ExtractInput
                Vec3d r(rsdata + k * 6);
                Vec3d t(tsdata + k * 6);
                Matx33d K(KData[0], 0, KData[2], 0, KData[1], KData[3], 1);
                Mat_<double> D(motionView.D.rows, 1, DData);
                Mat_<Vec2d> errPoint2D = errPoint2Ds.rowRange(row1, row2 = row1 + motionView.point2D.rows);
                Mat_<double> dpdr; if (dpdKDTs.empty() == false) dpdr = dpdKDTs.rowRange(row1 * 2, row2 * 2).colRange(k * 6, k * 6 + 3);
                Mat_<double> dpdt; if (dpdKDTs.empty() == false) dpdt = dpdKDTs.rowRange(row1 * 2, row2 * 2).colRange(k * 6 + 3, k * 6 + 6);
                Mat_<double> dpdK; if (dpdKDTs.empty() == false) dpdK = dpdKDTs.rowRange(row1 * 2, row2 * 2).colRange(nView * 2 * 3, nView * 2 * 3 + 4);
                Mat_<double> dpdD; if (dpdKDTs.empty() == false) dpdD = dpdKDTs.rowRange(row1 * 2, row2 * 2).colRange(nView * 2 * 3 + 4, dpdKDTs.cols);

                //2.2 CalcJacAndErr
                Mat_<double> dpdKDT;
                Mat_<Vec2d> point2DEva;
                cv::projectPoints(motionView.point3D, r, t, K, D, point2DEva, dpdKDTs.empty() ? noArray() : dpdKDT);
                errPoint2D = point2DEva - motionView.point2D;
                if (dpdr.empty() == false) dpdKDT.colRange(0, 3).copyTo(dpdr);
                if (dpdt.empty() == false) dpdKDT.colRange(3, 6).copyTo(dpdt);
                if (dpdK.empty() == false) dpdKDT.colRange(6, 10).copyTo(dpdK);
                if (dpdD.empty() == false) dpdKDT.colRange(10, 10 + D.rows).copyTo(dpdD);
                if (0)//for DEBUG
                {
                    cout << norm(point2DEva - motionView.point2D, errPoint2Ds.rowRange(row1, row2 = row1 + motionView.point2D.rows), NORM_INF) << "\t";
                    if (dpdr.empty() == false) cout << norm(dpdKDT.colRange(0, 3), dpdKDTs.rowRange(row1 * 2, row2 * 2).colRange(k * 6, k * 6 + 3), NORM_INF) << "\t";
                    if (dpdt.empty() == false) cout << norm(dpdKDT.colRange(3, 6), dpdKDTs.rowRange(row1 * 2, row2 * 2).colRange(k * 6 + 3, k * 6 + 6), NORM_INF) << "\t";
                    if (dpdK.empty() == false) cout << norm(dpdKDT.colRange(6, 10), dpdKDTs.rowRange(row1 * 2, row2 * 2).colRange(nView * 2 * 3, nView * 2 * 3 + 4), NORM_INF) << "\t";
                    if (dpdK.empty() == false) cout << norm(dpdKDT.colRange(10, 10 + D.rows), dpdKDTs.rowRange(row1 * 2, row2 * 2).colRange(nView * 2 * 3 + 4, dpdKDTs.cols), NORM_INF) << endl;
                }
            }
            return true;
        }
    };
};

int main(int argc, char** argv) { OptimizeKDRt::TestMe(argc, argv); return 0; }
int main1(int argc, char** argv) { OptimizeRt::TestMe(argc, argv); return 0; }
int main2(int argc, char** argv) { OptimizeKDRt::TestMe(argc, argv); return 0; }