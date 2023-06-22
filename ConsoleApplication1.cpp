
#include<cstdio>
#include<iostream>
#include <opencv2/opencv.hpp>  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<math.h>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
    if (argc == 1) {
        cout << argv[0] << endl;
        return 0;
    }
    if (argc == 2) {
        cout << argv[1] << endl;


    }
    if (argc == 3) {
        cout << argv[2] << endl;


    }
    Mat img, img_gray, thresh;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    img = imread(argv[1], 1);
    cvtColor(img, img_gray, CV_BGR2GRAY);
    threshold(img_gray, thresh, 230, 255, THRESH_BINARY);
    findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());

    int brain = -1;
    int temp = -1;

    for (int i = 0; i < contours.size(); i++) {
        float area=contourArea(contours[i]);
        if (temp < area) {
            brain = i;
            temp = area;

        }

    }
    drawContours(img, contours, brain, Scalar(255, 0, 0), -1,8);
    for (int  nrow = 0; nrow < img.rows; nrow++)
    {
        for (int  ncol = 0; ncol < img.cols; ncol++)
        {
            if (img.at<Vec3b>(nrow, ncol)[0] == 255 && img.at<Vec3b>(nrow, ncol)[1] == 0 && img.at<Vec3b>(nrow, ncol)[2] == 0) {

                img.at<Vec3b>(nrow, ncol)[1] = 255;
                img.at<Vec3b>(nrow, ncol)[2] = 255;

           }
            else {

                img.at<Vec3b>(nrow, ncol)[0] = 0;
                img.at<Vec3b>(nrow, ncol)[1] = 0;
                img.at<Vec3b>(nrow, ncol)[2] = 0;
            }
            
            
        }
     
    }

    Mat need;
    bitwise_and(img, imread(argv[1], 1), need);

   
    Mat img_need;
    need.copyTo(img_need);
    cvtColor(img_need, img_gray, CV_BGR2GRAY);
    threshold(img_gray, thresh, 230, 255, THRESH_BINARY);
    dilate(thresh, thresh, Mat());
    findContours(thresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
    Rect parameter;
    parameter=boundingRect(contours[0]);
    Moments M;
    M=moments(contours[0]);

    int o1 = M.m10 / M.m00;
    int o2 = M.m01 / M.m00;

    drawContours(img_need, contours, 0, Scalar(255, 255, 255), -1, 8);


    int minv = 100000000;
    double index = -1;

    for (int ang = 0; ang < 185; ang += 3) {
        int change = 0;
        cvtColor(img_need, img_gray, CV_BGR2GRAY);
        threshold(img_gray, thresh, 230, 255, THRESH_BINARY);
        ellipse(thresh, Point(o1, o2), Size(int(parameter.width / 2), int(parameter.height / 2) ), ang, 0, 360, Scalar(0), -1);
        for (size_t nrow = 0; nrow < thresh.rows; nrow++)
        {
            uchar* data = thresh.ptr<uchar>(nrow);
            for (size_t ncol = 0; ncol < thresh.cols; ncol++)
            {
                
                if (data[ncol] == 255) {
                    change += 1;
                }
            }
        }

        if (change < minv) {
            minv = change;
            index = ang;
        }
    }


  img = need;
    if (index > 90) {

        index = index - 180;
    }
    Mat rot_mat(2, 3, CV_32FC1);
    rot_mat =getRotationMatrix2D(Point(o1, o2), index, 1);
    warpAffine(img, img, rot_mat, Size(img.rows, img.cols));
    imwrite(argv[2], img);
    /*imshow("Point of Contours", img);
    waitKey(0);
   */
    
}

// 執行程式: Ctrl + F5 或 [偵錯] > [啟動但不偵錯] 功能表
// 偵錯程式: F5 或 [偵錯] > [啟動偵錯] 功能表

// 開始使用的提示: 
//   1. 使用 [方案總管] 視窗，新增/管理檔案
//   2. 使用 [Team Explorer] 視窗，連線到原始檔控制
//   3. 使用 [輸出] 視窗，參閱組建輸出與其他訊息
//   4. 使用 [錯誤清單] 視窗，檢視錯誤
//   5. 前往 [專案] > [新增項目]，建立新的程式碼檔案，或是前往 [專案] > [新增現有項目]，將現有程式碼檔案新增至專案
//   6. 之後要再次開啟此專案時，請前往 [檔案] > [開啟] > [專案]，然後選取 .sln 檔案
