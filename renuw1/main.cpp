#include<opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
std::vector<std::vector<std::string>> readCSV(const std::string& filename);
using namespace cv;
void Stringsplit(std::string str, const char split,  std::vector<std::string>& str_);
void chuli(Mat& img, float img_scale, std::string interpolation,
    int img_horizon, int img_vertical,
    std::string Rotation_center, float Rotation_angle);
std::vector<std::string> parseCSVLine(const std::string& line);
void chuli(Mat& img, Size img_size, std::string interpolation,
    int img_horizon, int img_vertical,
    std::string Rotation_center, float Rotation_angle);
double bijiao(double shu, Mat& img);
int main()
{
    const std::string filename = "./dataset/experiment1.csv";
    std::vector<std::vector<std::string>>csvData = readCSV(filename);
    std::string path = "./dataset/";
    std::string path_="./data_final/";
    for (auto iter = (csvData.begin()+1); iter != csvData.end(); ++iter)
    {
        if ((*iter)[1].find(",") != (*iter)[1].npos)
        {
            Mat img = imread(path + (*iter)[0], 1);
            std::vector<std::string> str_;
            Stringsplit((*iter)[1], ',', str_);
            chuli(img, Size(std::stoi(str_[0]), std::stoi(str_[1])), (*iter)[2], std::stoi((*iter)[3]), std::stoi((*iter)[4]), (*iter)[5], std::stof((*iter)[6]));
            namedWindow("img", WINDOW_FREERATIO);
            imshow("img", img);
            waitKey(0);
	    imwrite(path_+(*iter)[0],img);

        }
        else
        {
            
            std::cout << (*iter)[1];
            Mat img = imread(path + (*iter)[0], 1);
            
            chuli(img, std::stof((*iter)[1]), (*iter)[2], std::stoi((*iter)[3]), std::stoi((*iter)[4]), (*iter)[5], std::stof((*iter)[6]));
            cv::Mat m(img.cols, img.rows, img.type(), Scalar(255,255,255));
            //bitwise_or(m,img,img);
            
            namedWindow("img", WINDOW_FREERATIO);
            imshow("img", img);
            waitKey(0);
	    imwrite(path_+(*iter)[0],img);
        }
        
    }
}
void Stringsplit(std::string str, const char split,std::vector<std::string>& str_)
{
    std::istringstream iss(str);
    std::string token;
    while(std::getline(iss,token,split))
    {
        str_.push_back(token);
    }
}
std::vector<std::string> parseCSVLine(const std::string& line) {
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string cell;
    bool inQuotes = false;
    char c;

    while (ss.get(c)) {
        if (c == '"') {
            inQuotes = !inQuotes;
        }
        else if (c == ',' && !inQuotes) {
            result.push_back(cell);
            cell.clear();
        }
        else {
            cell += c;
        }
    }
    result.push_back(cell);  // 最后一个单元格

    return result;
}

// 读取CSV文件
std::vector<std::vector<std::string>> readCSV(const std::string& filename) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open the file " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<std::string> row = parseCSVLine(line);
        data.push_back(row);
    }

    file.close();
    return data;
}
double bijiao(double shu,Mat& img)
{
    if (shu > img.cols)
    {
        return img.cols;
    }
    else if (shu < 0)
    {
        return 0;
    }
    else
    {
        return shu;
    }
}
void chuli(Mat& img, float img_scale, std::string interpolation,
    int img_horizon, int img_vertical,
    std::string Rotation_center, float Rotation_angle)
{
    std::string flag_1 = "NEAREST";
    std::string flag_2 = "LINEAR";
    std::string flag_3 = "center";
    std::string flag_4 = "origin";
    int interpolation_int;
    Point2f Rotation_center_int;
    if (!interpolation.compare(flag_1))
    {
        interpolation_int=1;
    }
    if (!interpolation.compare(flag_2))
    {
        interpolation_int = 2;
    }
    resize(img, img, Size(), img_scale, img_scale, interpolation_int);
    if (!Rotation_center.compare(flag_3))
    {
        Rotation_center_int = Point2f(img.cols / 2+ img_horizon, img.rows / 2+img_vertical);
    }
    if (!Rotation_center.compare(flag_4))
    {
        Rotation_center_int = Point2f(0 + img_horizon,0+img_vertical);
    }
    Mat rotationMatrix = getRotationMatrix2D(Rotation_center_int, Rotation_angle, 1.0);
    Point2f img_zs = Point2f(0, 0);
    Point2f img_ys = Point2f(img.cols, 0);
    Point2f img_yx = Point2f(img.cols, img.rows);
    Point2f img_zx = Point2f(0, img.rows);
    Mat img_zs_mat = (Mat_<double>(3, 1) << img_zs.x, img_zs.y, 1);
    Mat img_ys_mat = (Mat_<double>(3, 1) << img_ys.x, img_ys.y, 1);
   
    Mat img_yx_mat = (Mat_<double>(3, 1) << img_yx.x, img_yx.y, 1);
    Mat img_zx_mat = (Mat_<double>(3, 1) << img_zx.x, img_zx.y, 1);
    Mat transationMatrix = (cv::Mat_<double>(2, 3) << 1, 0, img_horizon, 0, 1, img_vertical);
    warpAffine(img, img, transationMatrix, img.size());

    warpAffine(img, img, rotationMatrix,img.size());
    Mat img_zs_mat_ = transationMatrix * img_zs_mat;
    Mat img_ys_mat_= transationMatrix * img_ys_mat;
    Mat img_yx_mat_= transationMatrix * img_yx_mat;
    Mat img_zx_mat_= transationMatrix * img_zx_mat;
    Mat img_zs_mat_1= (Mat_<double>(3, 1) << bijiao(img_zs_mat_.at<double>(0,0),img), bijiao(img_zs_mat_.at<double>(1, 0),img), 1);
    Mat img_ys_mat_1 = (Mat_<double>(3, 1) << bijiao(img_ys_mat_.at<double>(0, 0),img), bijiao(img_ys_mat_.at<double>(1, 0),img), 1);
    Mat img_yx_mat_1 = (Mat_<double>(3, 1) << bijiao(img_yx_mat_.at<double>(0, 0),img), bijiao(img_yx_mat_.at<double>(1, 0),img), 1);
    Mat img_zx_mat_1 = (Mat_<double>(3, 1) << bijiao(img_zx_mat_.at<double>(0, 0),img), bijiao(img_zx_mat_.at<double>(1, 0),img), 1);
    Mat transformed_point_mat_1 = rotationMatrix * img_zs_mat_1;
    Mat transformed_point_mat_2 = rotationMatrix * img_ys_mat_1;
    Mat transformed_point_mat_3 = rotationMatrix * img_yx_mat_1;
    Mat transformed_point_mat_4 = rotationMatrix * img_zx_mat_1;
    std::vector<Point> points = { Point(transformed_point_mat_1.at<double>(0,0),transformed_point_mat_1.at<double>(1,0)),Point(transformed_point_mat_2.at<double>(0,0),transformed_point_mat_2.at<double>(1,0)), Point(transformed_point_mat_3.at<double>(0, 0),transformed_point_mat_3.at<double>(1,0)),Point(transformed_point_mat_4.at<double>(0, 0),transformed_point_mat_4.at<double>(1, 0))};
    Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC3);
    fillPoly(mask, points, Scalar(255, 255, 255));
    Mat gray_;
    cvtColor(mask, gray_, COLOR_BGR2GRAY);
    threshold(gray_, gray_, 5, 255, THRESH_BINARY_INV);
    Mat rect_kernel;
    rect_kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(gray_, gray_, rect_kernel);
    img.setTo(Scalar(255, 255, 255), gray_);

}
void chuli(Mat& img, Size img_size, std::string interpolation,
    int img_horizon, int img_vertical,
    std::string Rotation_center, float Rotation_angle)
{
    std::string flag_1 = "NEAREST";
    std::string flag_2 = "LINEAR";
    std::string flag_3 = "center";
    std::string flag_4 = "origin";
    int interpolation_int;
    Point2f Rotation_center_int;
    if (!interpolation.compare(flag_1))
    {
        interpolation_int = 1;
    }
    if (!interpolation.compare(flag_2))
    {
        interpolation_int = 2;
    }
    resize(img, img, img_size,0,0, interpolation_int);
    if (!Rotation_center.compare(flag_3))
    {
        Rotation_center_int = Point2f(img.cols / 2+ img_horizon, img.rows / 2+ img_vertical);
    }
    if (!Rotation_center.compare(flag_4))
    {
        Rotation_center_int = Point2f(0+ img_horizon, 0+ img_vertical);
    }
    Mat rotationMatrix = getRotationMatrix2D(Rotation_center_int, Rotation_angle, 1.0);
    Point2f img_zs = Point2f(0, 0);
    Point2f img_ys = Point2f(img.cols, 0);
    Point2f img_yx = Point2f(img.cols, img.rows);
    Point2f img_zx = Point2f(0, img.rows);
    Mat img_zs_mat = (Mat_<double>(3, 1) << img_zs.x, img_zs.y, 1);
    Mat img_ys_mat = (Mat_<double>(3, 1) << img_ys.x, img_ys.y, 1);

    Mat img_yx_mat = (Mat_<double>(3, 1) << img_yx.x, img_yx.y, 1);
    Mat img_zx_mat = (Mat_<double>(3, 1) << img_zx.x, img_zx.y, 1);
    Mat transationMatrix = (cv::Mat_<double>(2, 3) << 1, 0, img_horizon, 0, 1, img_vertical);
    warpAffine(img, img, transationMatrix, img.size());

    warpAffine(img, img, rotationMatrix, img.size());
    Mat img_zs_mat_ = transationMatrix * img_zs_mat;
    Mat img_ys_mat_ = transationMatrix * img_ys_mat;
    Mat img_yx_mat_ = transationMatrix * img_yx_mat;
    Mat img_zx_mat_ = transationMatrix * img_zx_mat;
    Mat img_zs_mat_1 = (Mat_<double>(3, 1) << bijiao(img_zs_mat_.at<double>(0, 0), img), bijiao(img_zs_mat_.at<double>(1, 0), img), 1);
    Mat img_ys_mat_1 = (Mat_<double>(3, 1) << bijiao(img_ys_mat_.at<double>(0, 0), img), bijiao(img_ys_mat_.at<double>(1, 0), img), 1);
    Mat img_yx_mat_1 = (Mat_<double>(3, 1) << bijiao(img_yx_mat_.at<double>(0, 0), img), bijiao(img_yx_mat_.at<double>(1, 0), img), 1);
    Mat img_zx_mat_1 = (Mat_<double>(3, 1) << bijiao(img_zx_mat_.at<double>(0, 0), img), bijiao(img_zx_mat_.at<double>(1, 0), img), 1);
    Mat transformed_point_mat_1 = rotationMatrix * img_zs_mat_1;
    Mat transformed_point_mat_2 = rotationMatrix * img_ys_mat_1;
    Mat transformed_point_mat_3 = rotationMatrix * img_yx_mat_1;
    Mat transformed_point_mat_4 = rotationMatrix * img_zx_mat_1;
    std::vector<Point> points = { Point(transformed_point_mat_1.at<double>(0,0),transformed_point_mat_1.at<double>(1,0)),Point(transformed_point_mat_2.at<double>(0,0),transformed_point_mat_2.at<double>(1,0)), Point(transformed_point_mat_3.at<double>(0, 0),transformed_point_mat_3.at<double>(1,0)),Point(transformed_point_mat_4.at<double>(0, 0),transformed_point_mat_4.at<double>(1, 0)) };
    Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC3);
    fillPoly(mask, points, Scalar(255, 255, 255));
    Mat gray_;
    cvtColor(mask, gray_, COLOR_BGR2GRAY);
    threshold(gray_, gray_, 5, 255, THRESH_BINARY_INV);
    Mat rect_kernel;
    rect_kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(gray_, gray_, rect_kernel);
    img.setTo(Scalar(255, 255, 255), gray_);

}
