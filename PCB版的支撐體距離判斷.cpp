#include<cstdio>
#include<iostream>
#include <opencv2/opencv.hpp>  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include <cmath>
#include<math.h>
using namespace cv;
using namespace std;

float** transpose(float** data, int n,int m) {
    float** A1T;
    A1T = new float* [n];
    for (int i = 0; i < n; i++)
    {
        A1T[i] = new float[m];
        for (int j = 0; j < m; j++) {

            A1T[i][j] = data[j][i];
        }
    }
    return A1T;

}


float** multiple(float** A, float** B,int a,int b,int c) {

    float** A1TA;
    A1TA = new float* [a];
    for (int i = 0; i < a; i++)
    {
        A1TA[i] = new float[c];
    }


    for (int i = 0; i < a; i++)
    {
        for (int j = 0; j < c; j++)
        {
            A1TA[i][j] = 0;
            for (int k = 0; k <b; k++)
            {
                A1TA[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return A1TA;


}


float ** gauss_jordan(float** r1, float** r2,int m) {
    float tmp;
    float** arr_1;
    arr_1 = new float* [m];
    for (int i = 0; i < m; i++)
    {
        arr_1[i] = new float[1];
    }

    //arr_1[0][0] = (r2[1][0] - r1[0][0] * r1[1][1] / r1[0][1]) / (r1[1][0] - r1[0][0] * r1[1][1] / r1[0][1]);
    //arr_1[1][0] = (r2[0][0] - r1[0][0] * arr_1[0][0]) / r1[0][1];
    arr_1[0][0] = ((r1[1][1] * r2[0][0]) - (r1[0][1] * r2[1][0])) / ((-1 * r1[1][0] * r1[0][1]) + (r1[0][0] * r1[1][1]));
    arr_1[1][0] = (r2[0][0] - arr_1[0][0] * r1[0][0]) / r1[0][1];
    return arr_1;
}

vector<float> getLinearEquation(vector <float> data) {
    vector<float> answer;
    
    float p1x= data[0];
    float p1y= data[1];
    float p2x= data[2];
    float p2y= data[3];

    float sign = 1;
    float a = p2y - p1y;
    if (a < 0) { 
        sign = -1;
        a = sign * a;
    }
    float b = sign * (p1x - p2x);
    float c = sign * (p1y * p2x - p1x * p2y);

    answer.push_back(a);
    answer.push_back(b);
    answer.push_back(c);
    return answer;

}
int main(int argc, char** argv)
{
    Mat img, gray, blurred, canny, dilation;
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
    cvtColor(img, gray, CV_BGR2GRAY);
    GaussianBlur(gray, blurred,Size(3,3), 0);
    Canny(blurred, canny, 150, 255);
    dilate(canny, dilation, Mat());


    vector<vector<float>>  axis_1;
    vector<vector<float>>  axis_2;
    bool flag = false;

    for (size_t nrow = 0; nrow < dilation.cols; nrow++)
    {
        
        for (size_t ncol = 0; ncol < dilation.rows; ncol++)
        {
            uchar* data = dilation.ptr<uchar>(ncol);
            if (data[nrow] == 255 && flag == false) {
                vector<float> temp;

                int convertrow = static_cast<int>(nrow);
                int convertcol = static_cast<int>(ncol);
                temp.push_back(float(convertrow));
                temp.push_back(float(convertcol));
                flag = true;
                axis_1.push_back(temp);
            }
            if (data[nrow] == 255 && flag == true) {
                vector<float> temp;
                int convertrow = static_cast<int>(nrow);
                int convertcol = static_cast<int>(ncol);
                temp.push_back(float(convertrow));
                temp.push_back(float(convertcol));
                if ((axis_1[0][1] - ncol) > -30) {
                    //circle(img, Point(nrow, ncol), 1, Scalar(0, 255, 0), -1);
                    axis_1.push_back(temp);
                }
                else {
                    axis_2.push_back(temp);
                }
            }
        }
    }
    int M1=axis_1.size();
    float** A1;
    A1 = new float* [M1];
    for (int i = 0; i < M1; i++)
    {
        A1[i] = new float[2];
        A1[i][0] = 1.0;
        A1[i][1] = axis_1[i][0];
    }
    float** B1;
    B1 = new float* [M1];
    for (int i = 0; i < M1; i++)
    {
        B1[i] = new float[1];
        B1[i][0] = axis_1[i][1];
    
    }
    float **A1T=transpose(A1, 2, M1);

   
    float **result1=multiple(A1T, A1, 2, M1,2);
    float **result2 = multiple(A1T, B1, 2, M1,1);

    float** paramter1 = gauss_jordan(result1, result2, 2);

    float x1, y1, x2, y2;
    vector<float> line1;
    vector<float> line1_paramter;
    x1 = axis_1[0][0];
    x2 = axis_1[axis_1.size() - 1][0];
    
    y1 = paramter1[1][0] * x1 + paramter1[0][0];
    y2 = paramter1[1][0] * x2 + paramter1[0][0];
    line1.push_back(x1);
    line1.push_back(y1);
    line1.push_back(x2);
    line1.push_back(y2);

    line1_paramter = getLinearEquation(line1);
    //cout << line1_paramter[0] << " " << line1_paramter[1] << "　" << line1_paramter[2] << endl;


    int M2 = axis_2.size();
    float** A2;
    A2 = new float* [M2];
    for (int i = 0; i < M2; i++)
    {
        A2[i] = new float[2];
        A2[i][0] = 1.0;
        A2[i][1] = axis_2[i][0];
    }
    float** B2;
    B2 = new float* [M2];
    for (int i = 0; i < M2; i++)
    {
        B2[i] = new float[1];
        B2[i][0] = axis_2[i][1];

    }
    float** A2T = transpose(A2, 2, M2);


    float** result3 = multiple(A2T, A2, 2, M2, 2);
    float** result4 = multiple(A2T, B2, 2, M2, 1);

    float** paramter2 = gauss_jordan(result3, result4, 2);

    
    vector<float> line2;
    vector<float> line2_paramter;
    x1 = axis_2[0][0];
    x2 = axis_2[axis_2.size() - 1][0];

    y1 = paramter2[1][0] * x1 + paramter2[0][0];
    y2 = paramter2[1][0] * x2 + paramter2[0][0];
    line2.push_back(x1);
    line2.push_back(y1);
    line2.push_back(x2);
    line2.push_back(y2);

    line2_paramter = getLinearEquation(line2);
    //cout << line2_paramter[0] << " " << line2_paramter[1] << "　" << line2_paramter[2] << endl;

    float m1 = -1 * line1_paramter[0] / line1_paramter[1];
    float m2 = -1 * line2_paramter[0] / line2_paramter[1];
    m1=atan(m1);
    float angletop = m1*(180 / 3.14) / 10;
    m2 = atan(m2);
    float angledown = m2 * (180 / 3.14) / 10;


    float angle = 90 - (angletop + angledown) / 2;
    float h = angle * (3.14 / 180);
    float m=tan(h);
    float x = img.rows/2;
    float y = img.cols/2;
    float b = -m * x + y;
       
    x1 = (-line1_paramter[1] * m * x + line1_paramter[1] * y + line1_paramter[2]) / (line1_paramter[0] + line1_paramter[1] * m);
    x1 = x1 * -1;
    y1 = ((-line1_paramter[0] * x1) - line1_paramter[2]) / line1_paramter[1];


    x2 = (-line2_paramter[1] * m * x + line2_paramter[1] * y + line2_paramter[2]) / (line2_paramter[0] + line2_paramter[1] * m);
    x2 = x2 * -1;
    y2 = ((-line2_paramter[0] * x2) - line2_paramter[2]) / line2_paramter[1];

    float answer=sqrt(((x1 - x2)* (x1 - x2)) + ((y1 - y2) * (y1 - y2)));

    cout << fixed <<setprecision(4) <<answer << endl;
  
}

