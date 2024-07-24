#include<opencv2/opencv.hpp>
#include<vector>
#include<iostream>
#include <filesystem>
#include<string>
using namespace cv;
using namespace std;
namespace fs = std::filesystem;
int main()
{
	fs::path folderPath = "./dataset";
	for (const auto& entry : fs::directory_iterator(folderPath))
	{
		Mat img = imread(entry.path(), 1);
		Mat img1 = imread("./dataset/20220823_153344_86_ID1587.jpg", 1);
		Mat img_;
		Mat img1_;
		cvtColor(img, img_, COLOR_BGR2GRAY);
		cvtColor(img1, img1_, COLOR_BGR2GRAY);
		double scale_factor1 = static_cast<double>(cv::mean(img_).val[0]) / cv::mean(img1_).val[0];
		Mat mask2 = imread("./2.jpg", 1);
		cvtColor(mask2, mask2, COLOR_BGR2GRAY);
		threshold(mask2, mask2, 100, 255, THRESH_BINARY_INV);
		int kernelsize_ = 7;
		Mat kernel_ = getStructuringElement(MORPH_RECT, Size(kernelsize_, kernelsize_));
		erode(mask2, mask2, kernel_);
		vector<Point> points = { Point{0,img.rows / 3 + 100},Point(img.cols,img.rows / 3 + 100),Point(img.cols,img.rows),Point(0,img.rows) };
		Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC3);
		fillPoly(mask, points, Scalar(255, 255, 255));
		bitwise_and(img, mask, img);
		cvtColor(img, img, COLOR_BGR2GRAY);
		bitwise_and(img, mask2, img);
		GaussianBlur(img, img, Size(9, 9), 0);
		medianBlur(img, img, 7);
		if (scale_factor1 > 1)
		{
			scale_factor1 = scale_factor1 * 1.2;
		}
		else
		{
			scale_factor1 = scale_factor1 * 0.9;
		}
		threshold(img, img, 130 * scale_factor1, 255, THRESH_BINARY);
		int kernelsize = 5;
		Mat kernel = getStructuringElement(MORPH_RECT, Size(kernelsize, kernelsize));
		morphologyEx(img, img, MORPH_OPEN, kernel);
		namedWindow("img5", WINDOW_FREERATIO);
		imshow("img5", img);
		waitKey(0);
		static int i=1;
		string str=to_string(i);
		imwrite("./datafinal/"+str+".jpg",img);
		i++;
	}
	
}
