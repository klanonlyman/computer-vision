#include "iostream"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "vector"

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/ml/ml.hpp>

#include "DetectRegions.h"          
#include "OCR.h"
using namespace std;
using namespace cv::ml;
using namespace cv;
Mat histeq(Mat in)
{
    Mat out(in.size(), in.type());
    if (in.channels() == 3) {
        Mat hsv;
        vector<Mat> hsvSplit;
        cvtColor(in, hsv, CV_BGR2HSV);
        split(hsv, hsvSplit);
        equalizeHist(hsvSplit[2], hsvSplit[2]);
        merge(hsvSplit, hsv);
        cvtColor(hsv, out, CV_HSV2BGR);
    }
    else if (in.channels() == 1) {
        equalizeHist(in, out);
    }

    return out;

}
bool verifySizes(RotatedRect mr) {

    float error = 0.4;
    //Spain car plate size: 52x11 aspect 4,7272
    float aspect = 4.7272;
    //Set a min and max area. All other patchs are discarded
    int min = 15 * aspect * 15; // minimum area
    int max = 125 * aspect * 125; // maximum area
    //Get only patchs that match to a respect ratio.
    float rmin = aspect - aspect * error;
    float rmax = aspect + aspect * error;

    int area = mr.size.height * mr.size.width;
    float r = (float)mr.size.width / (float)mr.size.height;
    if (r < 1)
        r = (float)mr.size.height / (float)mr.size.width;

    if ((area < min || area > max) || (r < rmin || r > rmax)) {
        return false;
    }
    else {
        return true;
    }

}

Mat contrastStretch1(cv::Mat srcImage)
{
    cv::Mat resultImage = srcImage.clone();
    int nRows = resultImage.rows;
    int nCols = resultImage.cols;
 
    if (resultImage.isContinuous()) {
        nCols = nCols * nRows;
        nRows = 1;
    }

   
    double pixMin, pixMax;
    cv::minMaxLoc(resultImage, &pixMin, &pixMax);
    std::cout << "min_a=" << pixMin << " max_b=" << pixMax << std::endl;

  
    for (int j = 0; j < nRows; j++) {
        uchar* pDataMat = resultImage.ptr<uchar>(j);
        for (int i = 0; i < nCols; i++) {
            pDataMat[i] = (pDataMat[i] - pixMin) *
                255 / (pixMax - pixMin);        //255/(pixMax - pixMin)是斜率 y=k(x-a)
        }
    }
    return resultImage;
}


vector<Plate> Mysegment(Mat input) {
    vector<Plate> output;
    Mat img_gray;
    cvtColor(input, img_gray, CV_BGR2GRAY);
    GaussianBlur(img_gray, img_gray, Size(3, 3), 0, 0);
    Mat img_sobel;
    Sobel(img_gray, img_sobel, CV_8U, 1, 0,3, 1, 0, BORDER_DEFAULT);
    Mat img_threshold;
    threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
    morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);
    morphologyEx(img_threshold, img_threshold, CV_MOP_OPEN, element);
    element = getStructuringElement(MORPH_RECT, Size(3, 3));
    vector< vector< Point> > contours;
    findContours(img_threshold,
        contours,
        CV_RETR_EXTERNAL,
        CV_CHAIN_APPROX_NONE); 
    vector<vector<Point> >::iterator itc = contours.begin();
    vector<RotatedRect> rects;
    while (itc != contours.end()) {
        RotatedRect mr = minAreaRect(Mat(*itc));
        if (!verifySizes(mr)) {
            itc = contours.erase(itc);
        }
        else {
            ++itc;
            rects.push_back(mr);
        }
    }
    cv::Mat result;
    input.copyTo(result);
    cv::drawContours(result, contours,
        -1, 
        cv::Scalar(255, 0, 0),
        1); 
    for (int i = 0; i < rects.size(); i++) {
        circle(result, rects[i].center, 3, Scalar(0, 255, 0), -1);
        float minSize = (rects[i].size.width < rects[i].size.height) ? rects[i].size.width : rects[i].size.height;
        minSize = minSize - minSize * 0.5;
        srand(time(NULL));
        Mat mask;
        mask.create(input.rows + 2, input.cols + 2, CV_8UC1);
        mask = Scalar::all(0);
        int loDiff = 30;
        int upDiff = 30;
        int connectivity = 4;
        int newMaskVal = 255;
        int NumSeeds = 10;
        Rect ccomp;
        int flags = connectivity + (newMaskVal << 8) + CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;
        for (int j = 0; j < NumSeeds; j++) {
            Point seed;
            seed.x = rects[i].center.x + rand() % (int)minSize - (minSize / 2);
            seed.y = rects[i].center.y + rand() % (int)minSize - (minSize / 2);
            //seed.x = rects[i].center.x;
            //seed.y = rects[i].center.y;
            circle(result, seed, 1, Scalar(0, 255, 255), -1);
            int area = floodFill(input, mask, seed, Scalar(255, 0, 0), &ccomp, Scalar(loDiff, loDiff, loDiff), Scalar(upDiff, upDiff, upDiff), flags);
        }
        vector<Point> pointsInterest;
        Mat_<uchar>::iterator itMask = mask.begin<uchar>();
        Mat_<uchar>::iterator end = mask.end<uchar>();
        for (; itMask != end; ++itMask)
            if (*itMask == 255)
                pointsInterest.push_back(itMask.pos());
        RotatedRect minRect = minAreaRect(pointsInterest);
        if (verifySizes(minRect)) {
            Point2f rect_points[4]; minRect.points(rect_points);
            for (int j = 0; j < 4; j++)
                line(result, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 0, 255), 1, 8);
            float r = (float)minRect.size.width / (float)minRect.size.height;
            float angle = minRect.angle;
            if (r < 1)
                angle = 90 + angle;
            if (angle >= 120) {
                angle = 0;
            }
            Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);
            Mat img_rotated;
            warpAffine(input, img_rotated, rotmat, input.size(), CV_INTER_CUBIC);
            Size rect_size = minRect.size;
            if (r < 1)
                swap(rect_size.width, rect_size.height);
            Mat img_crop;
            getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);

            Mat resultResized;
            resultResized.create(33, 144, CV_8UC3);
            resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);
            Mat grayResult;
            cvtColor(resultResized, grayResult, CV_BGR2GRAY);
            blur(grayResult, grayResult, Size(3, 3));
            grayResult = histeq(grayResult);

            output.push_back(Plate(grayResult, minRect.boundingRect()));
        }
    }
    return output;


}
vector<Plate> segment(Mat input) {
    vector<Plate> output;
    Mat img_gray;
    cvtColor(input, img_gray, CV_BGR2GRAY);
    blur(img_gray, img_gray, Size(5, 5));
   Mat img_sobel;
    Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    Mat img_threshold;
    threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
     Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
    morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);
   
    vector< vector< Point> > contours;
    findContours(img_threshold,
        contours, // a vector of contours
        CV_RETR_EXTERNAL, // retrieve the external contours
        CV_CHAIN_APPROX_NONE); // all pixels of each contours

    vector<vector<Point> >::iterator itc = contours.begin();
    vector<RotatedRect> rects;
    while (itc != contours.end()) {
        RotatedRect mr = minAreaRect(Mat(*itc));
        if (!verifySizes(mr)) {
            itc = contours.erase(itc);
        }
        else {
            ++itc;
            
            rects.push_back(mr);
        }
    }
    
   
    cv::Mat result;
    input.copyTo(result);
    cv::drawContours(result, contours,
        -1, // draw all contours
        cv::Scalar(255, 0, 0), // in blue
        1); // with a thickness of 1

    for (int i = 0; i < rects.size(); i++) {

        circle(result, rects[i].center, 3, Scalar(0, 255, 0), -1);
        float minSize = (rects[i].size.width < rects[i].size.height) ? rects[i].size.width : rects[i].size.height;
        minSize = minSize - minSize * 0.5;
        srand(time(NULL));
        Mat mask;
        mask.create(input.rows + 2, input.cols + 2, CV_8UC1);
        mask = Scalar::all(0);
        int loDiff = 30;
        int upDiff = 30;
        int connectivity = 4;
        int newMaskVal = 255;
        int NumSeeds = 10;
        Rect ccomp;
        int flags = connectivity + (newMaskVal << 8) + CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;
        for (int j = 0; j < NumSeeds; j++) {
            Point seed;
            seed.x = rects[i].center.x + rand() % (int)minSize - (minSize / 2);
            seed.y = rects[i].center.y + rand() % (int)minSize - (minSize / 2);
            circle(result, seed, 1, Scalar(0, 255, 255), -1);
            int area = floodFill(input, mask, seed, Scalar(255, 0, 0), &ccomp, Scalar(loDiff, loDiff, loDiff), Scalar(upDiff, upDiff, upDiff), flags);
        }
        vector<Point> pointsInterest;
        Mat_<uchar>::iterator itMask = mask.begin<uchar>();
        Mat_<uchar>::iterator end = mask.end<uchar>();
        for (; itMask != end; ++itMask)
            if (*itMask == 255)
                pointsInterest.push_back(itMask.pos());

        RotatedRect minRect = minAreaRect(pointsInterest);
        
         if (verifySizes(minRect)) {
            Point2f rect_points[4]; minRect.points(rect_points);
            for (int j = 0; j < 4; j++)
                line(result, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 0, 255), 1, 8);
            float r = (float)minRect.size.width / (float)minRect.size.height;
            float angle = minRect.angle;
            
            if (r < 1)
                angle = 90 + angle;
            Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);
            Mat img_rotated;
            warpAffine(input, img_rotated, rotmat, input.size(), CV_INTER_CUBIC);
            Size rect_size = minRect.size;
            if (r < 1)
                swap(rect_size.width, rect_size.height);
            Mat img_crop;
            getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);
             Mat resultResized;
            resultResized.create(33, 144, CV_8UC3);
            resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);
           Mat grayResult;
            cvtColor(resultResized, grayResult, CV_BGR2GRAY);
           blur(grayResult, grayResult, Size(3, 3));
             grayResult = histeq(grayResult);
            output.push_back(Plate(grayResult, minRect.boundingRect()));
        }
    }
    return output;
}
int main() {



    FileStorage fs;
    fs.open("..//SVM.xml", FileStorage::READ);
    Mat SVM_TrainingData;
    Mat SVM_Classes;
    fs["TrainingData"] >> SVM_TrainingData;
    fs["classes"] >> SVM_Classes;
    Ptr<SVM> SVM_params = SVM::create();
    SVM_params->setType(SVM::C_SVC);
    SVM_params->setKernel(SVM::LINEAR);
    SVM_params->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 1000, 0.01));


    int Feature_n = 111;
   
    cout << SVM_TrainingData.size() << endl;
    PCA pca(SVM_TrainingData, Mat(), CV_PCA_DATA_AS_ROW, Feature_n);
    Mat PCA_DATA(111, Feature_n, CV_32FC1);



    for (int index = 0; index < 1; index += 1) {
        Mat dst = pca.project(SVM_TrainingData).row(index);
        PCA_DATA.row(index) = dst;
        
    }




    cout << PCA_DATA.size() << endl;

    SVM_params->train(SVM_TrainingData, ROW_SAMPLE, SVM_Classes);
    OCR ocr("..//OCR.xml");
    ocr.saveSegments = true;
    ocr.DEBUG = false;

    vector<string> imgname = { "..//2715DTZ.jpg", "..//3028BYS.jpg", "..//3154FFY.jpg", "..//3266CNT.jpg","..//3732FWW.jpg","..//5445BSX.jpg",
        "..//7215BGN.jpg","..//8995CCN.jpg","..//9588DWV.jpg","..//9773BNB.jpg","..//DSC_0562.jpg","..//DSC_0566.jpg" };
    map<int, int> maps;
    double START, END;
    START = clock();
    for (int name_index = 0; name_index < imgname.size(); name_index++) {
        cout << imgname[name_index] << " ";
        Mat input_image = imread(imgname[name_index], 1);
        vector<Plate> posible_regions = Mysegment(input_image);
        vector<Plate> plates;
        cout << posible_regions.size() << " ";
        for (int i = 0; i < posible_regions.size(); i++)
        {
            Mat img = posible_regions[i].plateImg;
            // imshow("red", img);
            // cvWaitKey(0);
            Mat p = img.reshape(1, 1);
            //p = pca.project(p);
            p.convertTo(p, CV_32FC1);
            int response = (int)SVM_params->predict(p);
            if (response == 1) {
                plates.push_back(posible_regions[i]);
            }
        }
       // cout << plates.size() << endl;
        if (plates.size() == 0) {
            posible_regions = segment(input_image);
            for (int i = 0; i < posible_regions.size(); i++)
            {
                // imshow("Detected", img);
                // cvWaitKey(0);
                Mat img = posible_regions[i].plateImg;
                Mat p = img.reshape(1, 1);
                //p = pca.project(p);//
                p.convertTo(p, CV_32FC1);
                int response = (int)SVM_params->predict(p);
                if (response == 1) {
                    plates.push_back(posible_regions[i]);
                }
            }
        }
        maps[plates.size()] += 1;




        for (int i = 0; i < plates.size(); i++) {
            Plate plate = plates[i];
            // imshow("Plate Detected", plates[i].plateImg);
            // cvWaitKey(0);
            string plateNumber = ocr.run(&plate);
            string licensePlate = plate.str();
            cout << "================================================\n";
            cout << "License plate number: " << licensePlate << "\n";
            cout << "================================================\n";
            rectangle(input_image, plate.position, Scalar(0, 0, 200));
            putText(input_image, licensePlate, Point(plate.position.x, plate.position.y), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 200), 2);
        }

    }

    

    END = clock();
    cout << endl << "進行運算所花費的時間：" << (END - START) / CLOCKS_PER_SEC<<endl;
    auto iter = maps.begin();
    while (iter != maps.end()) {
        cout << "[" << iter->first << ","
            << iter->second << "]\n";
        ++iter;

   }
	return 0;

    }
    