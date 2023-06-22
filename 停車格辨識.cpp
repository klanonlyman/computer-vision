// ConsoleApplication1.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//


#include<cstdio>
#include<iostream>
#include <opencv2/opencv.hpp>  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include <cmath>
using namespace cv;
using namespace std;
vector<int> cross_point(vector<int> line1, vector<int> line2) {
    vector<int > xy(2);
    int x1 = line1[0];
    int y1 = line1[1];
    int x2 = line1[2];
    int y2 = line1[3];

    int x3 = line2[0];
    int y3 = line2[1];
    int x4 = line2[2];
    int y4 = line2[3];

    
    float k1 = (y2-y1)*1.0 /(x2-x1)*1.0;
    if ((x2 - x1) == 0) {
       k1 = (y2 - y1) * 1.0;
    }
    

    float b1 = y1 * 1.0 - x1 * k1 * 1.0;
    float b2;
    float k2;
    if ((x4 - x3) == 0) {
      
        k2 = NAN;
        b2 = 0.0;
     }
           
    else {
      
        k2 = (y4 - y3) * 1.0 / (x4 - x3)*1.0;
        b2 = y3 * 1.0 - x3 * k2 * 1.0;
    }
    float x;
    if (k2!=k2) {
        
        x = x3;
        
     }
    else {
        
        x = (b2 - b1) * 1.0 / (k1 - k2)*1.0;
        
    }
    xy[0] =int(x);
    float temp1= float(k1) * x * 1.0;
    float temp2 = b1 * 1.0;
    float temp = temp1 + temp2;
   
    xy[1] = int(temp);
    
    return xy;




}

double calculateDistance(vector<int> x,vector<int> y)
{

    float temp=0;
    for (int i = 0; i < x.size(); i++) {

        temp += pow((float(x[i]) - float(y[i])),2);

    }

    return sqrt(temp);
}
void process(int* x1, int* x2) {
    
    if (*x1 < 0) {*x1 = 0;}
    if (*x2 < 0) {*x2 = 0;}
    if (*x1 > *x2) {
        int temp = *x1;
        *x1 = *x2;
        *x2 = temp;
    }


}

float slopee(int x1, int y1, int x2, int y2) {

    if ((x2 - x1) == 0) {
        return 0;
    }
    
    float x = float(y2 - y1) / float(x2 - x1);
    if (x < 0) {
        x = x * -1.0;
    }
    return x;


}

double average(vector<int> x, int len)
{
    double sum = 0;
    for (int i = 0; i < len; i++) // 求和
        sum += x[i];
    return sum / len; // 得到平均值
}
double variance(vector<int> x, int len)
{
    double sum = 0;
    double avg = average(x, len);
    for (int i = 0; i < len; i++) // 求和
        sum += pow(x[i] - avg, 2);
    return sum / len; // 得到方差
}
double standardDev(vector<int> x, int len)
{
    double var = variance(x, len);
    return sqrt(var); // 得到标准差
}


int main(int argc, char** argv)
{
    Mat img,gray_img, blur_img, bi_img,after, lineimg;
    vector<Vec2f> lines;
    vector<vector<int>> possible;
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

    img = imread(argv[1], 1);
    int w = img.cols;
    int h = img.rows;
    cvtColor(img, gray_img, CV_BGR2GRAY);
    //cout << w << " " << h << endl;
    medianBlur(gray_img, blur_img, 3);
    adaptiveThreshold(blur_img, bi_img, 255, 1, 1, 11, 3);
    dilate(bi_img, after, Mat());
    erode(after, after, Mat());

    erode(after, after, Mat());
    dilate(after, after, Mat());

    dilate(after, after, Mat());
    erode(after, after, Mat());

    medianBlur(after, after, 3);
    Canny(after, lineimg, 150, 255);

    HoughLines(lineimg, lines, 1, CV_PI / 180, 118);
    
    for (int i = 0; i < lines.size(); i++) {
        vector <int> pt1_l(2);
        vector <int> pt2_1(2);
        float rho_l = lines[i][0];
        float theta_l = lines[i][1];
        float a_l = cos(theta_l);
        float b_l = sin(theta_l);
        float x0_l = a_l * rho_l;
        float y0_l = b_l * rho_l;

        pt1_l[0]=int(x0_l + w * (-b_l));
        pt1_l[1]=int(y0_l + h * (a_l));

        pt2_1[0]=int(x0_l - w * (-b_l));
        pt2_1[1]=int(y0_l - h * (a_l));
       
        
        int x1 = pt1_l[0];
        int y1 = pt1_l[1];
        int x2 = pt2_1[0];
        int y2 = pt2_1[1];
        
        process(&x1, &x2);
        process(&y1, &y2);
        //cout << x1 <<" " << y1 << " " << x2 << " " << y2 << endl;
        if (slopee(pt1_l[0], pt1_l[1], pt2_1[0], pt2_1[1]) <= 0.01) {
            vector <int> localMAT;
            for (size_t nrow = y1; nrow <y2; nrow++)
            {
                uchar* data = gray_img.ptr<uchar>(nrow);
                for (size_t ncol = x1; ncol < x2+1; ncol++)
                {
                    localMAT.push_back(int(data[ncol]));
                }
            }
            double std=standardDev(localMAT, localMAT.size());
            
            if (std < 40) {
                vector<int> t{ x1, y1, x2, y2 };
                
                possible.push_back(t);
            }
        }
        
    }

   


    map<int,int> disuse;
    vector<vector<int>> accumulates;


    for (int i = 0; i < possible.size(); i++) {
        if (disuse[i] == 1) {
            continue;
        }
        vector<int> temp =possible[i];
        vector<int> accumulate{ 0,0,0,0 };
        int number = 0;

        for (int j = 0; j < possible.size(); j++) {
            vector<int> temp2 = possible[j];
            double dist=calculateDistance(temp, temp2);
            
            if (dist < 75) {
                disuse[j] = 1;
                accumulate[0] += temp2[0];
                accumulate[1] += temp2[1];
                accumulate[2] += temp2[2];
                accumulate[3] += temp2[3];
                number += 1;
            }



        }
        accumulate[0] /= number;
        accumulate[1] /= number;
        accumulate[2] /= number;
        accumulate[3] /= number;
        //cout << accumulate[0] <<endl;
        accumulates.push_back(accumulate);

    }

    vector<vector<int>> vertical;
    vector<vector<int>> horizontal;

    for (int i = 0; i < accumulates.size(); i++) {
        int x = abs(accumulates[i][0] - accumulates[i][2]);
        int  y = abs(accumulates[i][1] - accumulates[i][3]);
        if (x > y) { horizontal.push_back(accumulates[i]);}
        else { vertical.push_back(accumulates[i]); }
    }

    int horizontal_temp = horizontal.size();
    int vertical_temp = vertical.size()-1;
    
    if (horizontal_temp > vertical_temp) {
        horizontal_temp = horizontal.size()- 1;
        vertical_temp = vertical.size();
        int temp = vertical_temp;
        vertical_temp = horizontal_temp;
        horizontal_temp = temp;
    }
    float points = horizontal_temp / 2.0 - int(horizontal_temp / 2);
    if (points >= 0.5) {
        horizontal_temp = int(horizontal_temp / 2.0) + 1;
    }
    else {
        horizontal_temp = int(horizontal_temp / 2.0);
    }
    //cout << horizontal_temp << " " << vertical_temp << endl;
    int answer1 = horizontal_temp * 2 * vertical_temp;
    //cout << answer1 << endl;


    vector<int> horizontal_index;
    vector<int> vertical_index;
    map<int, int> horizontal_map;
    map<int, int> vertical_map;
    for (int i = 0; i < vertical.size(); i++) {
        int index = -1;
        for (int j = 0; j < vertical.size(); j++) {
            if (vertical_map[j] == 1) {
                continue;
            }
            if (index == -1) {
                index = j;
            }
            if (vertical[index][0] > vertical[j][0]) {
                index = j;
            }


        }
        vertical_map[index] = 1;
        vertical_index.push_back(index);

    }

    for (int i = 0; i < horizontal.size(); i++) {
        int index = -1;
        for (int j = 0; j < horizontal.size(); j++) {
            if (horizontal_map[j] == 1) {
                continue;
            }
            if (index == -1) {
                index = j;
            }
            if (horizontal[index][1] > horizontal[j][1]) {
                index = j;
            }


        }
        horizontal_map[index] = 1;
        horizontal_index.push_back(index);

    }
   // cout << horizontal_index.size()<<" " << vertical_index.size() << endl;
    int number = 0;
    if (horizontal_index.size() < vertical_index.size()) {

        for (int i = 0; i < vertical_index.size()-1; i++) {
            int index1 = vertical_index[i];
            int index2 = vertical_index[i+1];
            
          for (int j = 0; j < horizontal_index.size(); j++) {


                vector<int> left = cross_point(vertical[index1], horizontal[j]);
                vector<int> right = cross_point(vertical[index2], horizontal[j]);
                //cout << left[0] << ' ' << left[1] << endl;
                int width = calculateDistance(left, right);
                
                vector<int> left_top{ vertical[index1][0], vertical[index1][1] };
                vector<int> left_down{ vertical[index1][2], vertical[index1][3] };
                int hight_top = calculateDistance(left_top,left);
                int hight_down = calculateDistance(left_down,left);


               vector <int> localMAT;
                for (size_t nrow = left_top[1]; nrow < left_top[1]+ hight_top; nrow++)
                {
                   uchar* data = gray_img.ptr<uchar>(nrow);
                    for (size_t ncol = left_top[0]; ncol < left_top[0] + width; ncol++)
                    {
                        localMAT.push_back(int(data[ncol]));
                    }
                }
                double std = standardDev(localMAT, localMAT.size());
                //cout << std << endl;
                if (std > 30) {
                    number += 1;
                }
                
 
                vector <int> localMAT2;
                for (size_t nrow = left[1]; nrow < left[1] + hight_down; nrow++)
                {
                    uchar* data = gray_img.ptr<uchar>(nrow);
                    for (size_t ncol = left[0]; ncol < left[0] + width; ncol++)
                    {
                        localMAT2.push_back(int(data[ncol]));
                    }
                }
               std = standardDev(localMAT2, localMAT2.size());

                if (std > 30) {
                    number += 1;
                }
              
            }



        }


    }
    else {
        //cout << horizontal_index.size() << " " << vertical_index.size() << endl;

        for (int i = 0; i< horizontal.size(); i++) {

            horizontal[i][2] -= 1;
        }

        for (int i = 0; i < horizontal_index.size()-1; i++) {
            int index1 = horizontal_index[i];
            int index2 = horizontal_index[i + 1];

            for (int j = 0; j < vertical_index.size(); j++) {

                //cout <<"X:" << horizontal[index1][0] << " " << horizontal[index1][1] <<" " << horizontal[index1][2] << " " << horizontal[index1][3] << endl;
                //cout << "Y:" << vertical[j][0] << " " << vertical[j][1] << " " << vertical[j][2] << " " << vertical[j][3] << endl;
                vector<int> left = cross_point(horizontal[index1], vertical[j]);
                vector<int> right = cross_point(horizontal[index2], vertical[j]);
                //cout <<"left:" << left[0] << ' ' << left[1] << endl;
                int width = calculateDistance(left, right);

                vector<int> left_top{ horizontal[index1][0], horizontal[index1][1] };
                vector<int> left_down{ horizontal[index1][2], horizontal[index1][3] };
                int hight_top = calculateDistance(left_top, left);
                int hight_down = calculateDistance(left_down, left);
                //cout << width << " " << hight_top << endl;
                vector <int> localMAT;
                for (size_t nrow = left_top[1]; nrow < left_top[1] + width; nrow++)
                {
                    uchar* data = gray_img.ptr<uchar>(nrow);
                    for (size_t ncol = left_top[0]; ncol < left_top[0] + hight_top; ncol++)
                    {
                        localMAT.push_back(int(data[ncol]));
                    }
                }
                double std = standardDev(localMAT, localMAT.size());
                //cout << std << endl;
                if (std > 30) {
                    number += 1;
                }


               vector <int> localMAT2;
                for (size_t nrow = left[1]; nrow < left[1] + width; nrow++)
                {
                    uchar* data = gray_img.ptr<uchar>(nrow);
                    for (size_t ncol = left[0]; ncol < left[0] + hight_down; ncol++)
                    {
                        localMAT2.push_back(int(data[ncol]));
                    }
                }
                std = standardDev(localMAT2, localMAT2.size());

                if (std > 30) {
                    number += 1;
                }

            }



        }



    }

    cout << number << " " << answer1 << endl;
    /*namedWindow("HSV", WINDOW_AUTOSIZE);
    imshow("HSV", after);

    waitKey(0);*/
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
