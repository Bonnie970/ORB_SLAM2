#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "Frame.h"
#include "ORBVocabulary.h"

using namespace std;

int main ( int argc, char** argv )
{
    //-- 读取图像
    // cv::Mat img_1 = cv::imread ( "/guanqing_ORB_SLAM2/test_img.png" );
    string f1 = "/guanqing_ORB_SLAM2/1540019057_1010_00084380_crop.png";
    string f2 = "/guanqing_ORB_SLAM2/1540019057_1010_00084381_crop.png";
    cv::Mat img_1 = cv::imread (f1); 
    cv::Mat img_2 = cv::imread (f2); 
    cv::Mat mImGray1 = img_1;
    cv::Mat mImGray2 = img_2;
    cv:: Mat output_cv, outimg1, outimg2, outimg3, outimg4;//输出图像
    cvtColor(mImGray1,mImGray1, cv::COLOR_RGB2GRAY);//转换为灰度图
    cvtColor(mImGray2,mImGray2, cv::COLOR_RGB2GRAY);//转换为灰度图
    std::vector<cv::KeyPoint> keypoints_cv,keypoints_1,keypoints_2;
    cv::Mat descriptors_cv,descriptors_1,descriptors_2;

    //opencv中接口函数
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    detector->detect ( mImGray1,keypoints_cv );
    descriptor->compute ( mImGray1, keypoints_cv, descriptors_cv );
    cv::drawKeypoints( img_1, keypoints_cv, output_cv, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
    cv::imwrite("/guanqing_ORB_SLAM2/opencv_keypoints.png", output_cv);

    //调用ORB SLAM中特征提取函数
    ORB_SLAM2::ORBextractor* mpIniORBextractor;
    mpIniORBextractor = new ORB_SLAM2::ORBextractor(500,1.2,8,20,10);
    (*mpIniORBextractor)(mImGray1,cv::Mat(),keypoints_1,descriptors_1 ) ;
    (*mpIniORBextractor)(mImGray2,cv::Mat(),keypoints_2,descriptors_2 ) ;
    cv::drawKeypoints( img_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
    cv::imwrite("/guanqing_ORB_SLAM2/slam_keypoints.png",outimg1);
    cv::drawKeypoints( img_2, keypoints_2, outimg2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
    cv::imwrite("/guanqing_ORB_SLAM2/slam_keypoints_2.png",outimg2);   


    //调用Frame封装的ORB SLAM中特征提取函数
    // camera parameter uses ./Examples/Monocular/KITTI00-02.yaml
    // cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    // K.at<float>(0,0) = 718.856;
    // K.at<float>(1,1) = 718.856;
    // K.at<float>(0,2) = 607.1928;
    // K.at<float>(1,2) = 185.2157;
    // cv::Mat DistCoef(4,1,CV_32F);
    // DistCoef.at<float>(0) = 0.0;
    // DistCoef.at<float>(1) = 0.0;
    // DistCoef.at<float>(2) = 0.0;
    // DistCoef.at<float>(3) = 0.0;
    // float bf = 0.0;
    // float thDepth = 0.0;
    // double timestamp_dummy = 1;

    // ORB_SLAM2::Frame frame1(mImGray1, timestamp_dummy, mpIniORBextractor, static_cast<ORB_SLAM2::ORBVocabulary*>(NULL), K, DistCoef, bf, thDepth);
    // ORB_SLAM2::Frame frame2(mImGray2, timestamp_dummy, mpIniORBextractor, static_cast<ORB_SLAM2::ORBVocabulary*>(NULL), K, DistCoef, bf, thDepth);
    
    // cv::drawKeypoints( img_1, frame1.mvKeysUn, outimg3, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
    // cv::imwrite("/guanqing_ORB_SLAM2/slam_keypoints_3.png",outimg3);
    // cv::drawKeypoints( img_2, frame2.mvKeysUn, outimg4, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
    // cv::imwrite("/guanqing_ORB_SLAM2/slam_keypoints_4.png",outimg4);   

    //cv bfmatcher 指定L1或L2距离
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    cv::Mat result;
    //-- Filter matches using the Lowe's ratio test https://docs.opencv.org/3.4/d7/dff/tutorial_feature_homography.html
    const float max_distance = 200;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance < max_distance)
        {
            good_matches.push_back(matches[i]);
        }
    }
    //从两个图像中绘制找到的关键点匹配
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, result, cv::Scalar::all(-1), cv::Scalar(0,255,0));
    cv::imwrite("/guanqing_ORB_SLAM2/slam_bfmatch.png", result);  

    // homography  
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        points1.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
        points2.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
    }
    cv::Mat H = cv::findHomography( points1 , points2, cv::RANSAC); 
    cv::Mat warp_img_1; 
	cv::warpPerspective(img_1, warp_img_1, H, img_1.size());
    cv::imwrite("/guanqing_ORB_SLAM2/slam_bfmatch_warp.png", warp_img_1);  

    return 0;
}
