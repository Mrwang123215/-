#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <filesystem>
#include<string>
#include<vector>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
namespace fs = std::filesystem;
const float kRatioThresh = 0.7f;
int main() {
     //查看是否正确输入图像
    //读取图像
    fs::path folderPath = "./dataset/archive";
    fs::path folderPath_ = "./dataset/template";
    for (const auto& entry : fs::directory_iterator(folderPath))
    {
    	cv::Mat img1_1 = cv::imread("./dataset/template/template_1.jpg", cv::IMREAD_GRAYSCALE);
    	cv::Mat img2 = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE);
    // 创建 SIFT 检测器
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
 
    vector<cv::KeyPoint> keypoints1_1, keypoints2_1;
    cv::Mat descriptors1_1, descriptors2_1;
 
    sift -> detectAndCompute(img1_1,cv::Mat(),keypoints1_1, descriptors1_1);
    sift -> detectAndCompute(img2, cv::Mat(),keypoints2_1,descriptors2_1);
 
    //创建FLANN匹配器
    cv::Ptr<cv::FlannBasedMatcher> flannmatcher = cv::FlannBasedMatcher::create();
    //匹配关键点
    vector<vector<cv::DMatch>> matches_1;
    flannmatcher -> knnMatch(descriptors1_1, descriptors2_1, matches_1, 2);
 
    //根据最近邻次近邻距离比来筛选特征点
    vector<cv::DMatch> good_matches_1;
    for(size_t i=0;i<matches_1.size();i++)
    {
        if(matches_1[i][0].distance < kRatioThresh*matches_1[i][1].distance)
        {
            good_matches_1.push_back(matches_1[i][0]);
        }
    }
    cv::Mat img1_2 = cv::imread("./dataset/template/template_2.jpg", cv::IMREAD_GRAYSCALE);
    //cv::Mat img2 = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE);
    // 创建 SIFT 检测器
    //cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
 
    vector<cv::KeyPoint> keypoints1_2, keypoints2_2;
    cv::Mat descriptors1_2, descriptors2_2;
 
    sift -> detectAndCompute(img1_2,cv::Mat(),keypoints1_2, descriptors1_2);
    sift -> detectAndCompute(img2, cv::Mat(),keypoints2_2,descriptors2_2);
 
    //创建FLANN匹配器
    //cv::Ptr<cv::FlannBasedMatcher> flannmatcher = cv::FlannBasedMatcher::create();
    //匹配关键点
    vector<vector<cv::DMatch>> matches_2;
    flannmatcher -> knnMatch(descriptors1_2, descriptors2_2, matches_2, 2);
 
    //根据最近邻次近邻距离比来筛选特征点
    vector<cv::DMatch> good_matches_2;
    for(size_t i=0;i<matches_2.size();i++)
    {
        if(matches_2[i][0].distance < kRatioThresh*matches_2[i][1].distance)
        {
            good_matches_2.push_back(matches_2[i][0]);
        }
    }
     cv::Mat img1_3 = cv::imread("./dataset/template/template_1.jpg", cv::IMREAD_GRAYSCALE);
    //cv::Mat img2 = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE);
    // 创建 SIFT 检测器
    //cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
 
    vector<cv::KeyPoint> keypoints1_3, keypoints2_3;
    cv::Mat descriptors1_3, descriptors2_3;
 
    sift -> detectAndCompute(img1_3,cv::Mat(),keypoints1_3, descriptors1_3);
    sift -> detectAndCompute(img2, cv::Mat(),keypoints2_3,descriptors2_3);
 
    //创建FLANN匹配器
    //cv::Ptr<cv::FlannBasedMatcher> flannmatcher = cv::FlannBasedMatcher::create();
    //匹配关键点
    vector<vector<cv::DMatch>> matches_3;
    flannmatcher -> knnMatch(descriptors1_3, descriptors2_3, matches_3, 2);
 
    //根据最近邻次近邻距离比来筛选特征点
    vector<cv::DMatch> good_matches_3;
    for(size_t i=0;i<matches_3.size();i++)
    {
        if(matches_3[i][0].distance < kRatioThresh*matches_3[i][1].distance)
        {
            good_matches_3.push_back(matches_1[i][0]);
        }
    } 
    int w=0;
    static int m=1;
		if(good_matches_1.size()>good_matches_2.size()&&good_matches_1.size()>good_matches_3.size()&&good_matches_1.size()>7)
    	{
    		 cv::Mat img_matches;
    cv::drawMatches(img1_1,keypoints1_1, img2, keypoints2_1, good_matches_1, img_matches);
 
    //显示匹配结果
    cv::imshow("Matches",img_matches);
    cv::waitKey(10);
    string str=to_string(m);
	imwrite("./wuyu/"+str+".jpg",img_matches);
	m++;
    	}
    	else if(good_matches_2.size()>good_matches_1.size()&&good_matches_2.size()>good_matches_3.size()&&good_matches_2.size()>7)
    	{
    		 cv::Mat img_matches;
    cv::drawMatches(img1_2,keypoints1_2, img2, keypoints2_2, good_matches_2, img_matches);
 
    //显示匹配结果
    cv::imshow("Matches",img_matches);
    cv::waitKey(10);
    string str=to_string(m);
	imwrite("./wuyu/"+str+".jpg",img_matches);
	m++;
    	}
    	else if(good_matches_3.size()>good_matches_2.size()&&good_matches_3.size()>good_matches_1.size()&&good_matches_3.size()>7)
    	{
    		 cv::Mat img_matches;
    cv::drawMatches(img1_3,keypoints1_3, img2, keypoints2_3, good_matches_3, img_matches);
    //显示匹配结果
    cv::imshow("Matches",img_matches);
    cv::waitKey(10);
    string str=to_string(m);
	imwrite("./wuyu/"+str+".jpg",img_matches);
	m++;
    }
    }
 
    return 0;
}
