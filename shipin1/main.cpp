#include<opencv2/opencv.hpp>
using namespace cv;
int main()
{
	VideoCapture cap("./cpp.mp3");
	Mat img;
	while(cap.read(img))
	{
		imshow("img",img);
		waitKey(10);
	}
}
