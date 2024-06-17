#ifndef __OPENCV_CONVERSIONS_HPP__
#define __OPENCV_CONVERSIONS_HPP__

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include "vio_msgs/msg/d_matches.hpp"
#include "vio_msgs/msg/key_points.hpp"
#include "std_msgs/msg/Header.hpp"

class OpenCVConversions
{
private:
    OpenCVConversions() {}
public:
    static vio_msgs::msg::KeyPoints toRosKeyPoints(const std::vector<cv::KeyPoint> &keypoints);
    static std::vector<cv::KeyPoint> toCvKeyPoints(const vio_msgs::msg::KeyPoints &keypoints);

    static vio_msgs::msg::DMatches toRosDMatches(const std::vector<cv::DMatch> &matches);
    static std::vector<cv::DMatch> toCvDMatches(const vio_msgs::msg::DMatches &matches);

    static sensor_msgs::msg::Image toRosImage(const cv::Mat &image, std::string encoding = "mono8");
    static cv::Mat toCvImage(const sensor_msgs::msg::Image &image, std::string encoding = "mono8");
};

static vio_msgs::msg::KeyPoints OpenCVConversions::toRosKeyPoints(const std::vector<cv::KeyPoint> &keypoints)
{
    std::vector<float> angle, response, size;
    std::vector<int> class_id, octave, pt_x, pt_y;

    for (const cv::KeyPoint &kp : keypoints)
    {
        angle.push_back(kp.angle);
        response.push_back(kp.response);
        size.push_back(kp.size);
        class_id.push_back(kp.class_id);
        octave.push_back(kp.octave);
        pt_x.push_back(kp.pt.x);
        pt_y.push_back(kp.pt.y);
    }

    vio_msgs::msg::KeyPoints ros_keypoints;
    ros_keypoints.angle.data = angle;
    ros_keypoints.response.data = response;
    ros_keypoints.size.data = size;
    ros_keypoints.class_id.data = class_id;
    ros_keypoints.octave.data = octave;
    ros_keypoints.pt_x.data = pt_x;
    ros_keypoints.pt_y.data = pt_y;

    return ros_keypoints;
}

static std::vector<cv::KeyPoint> OpenCVConversions::toCvKeyPoints(const vio_msgs::msg::KeyPoints &keypoints)
{
    std::vector<cv::KeyPoint> cv_keypoints;
    for (size_t i = 0; i < keypoints.angle.data.size(); i++)
    {
        cv::KeyPoint kp(
            keypoints.pt_x.data[i],
            keypoints.pt_y.data[i],
            keypoints.size.data[i],
            keypoints.angle.data[i],
            keypoints.response.data[i],
            keypoints.octave.data[i],
            keypoints.class_id.data[i]
        );

        cv_keypoints.push_back(kp);
    }

    return cv_keypoints;
}

static vio_msgs::msg::DMatches OpenCVConversions::toRosDMatches(const std::vector<cv::DMatch> &matches)
{
    std::vector<int> img_idx, query_idx, train_idx;
    std::vector<float> distance;

    for (const cv::DMatch &match : matches)
    {
        img_idx.push_back(match.imgIdx);
        query_idx.push_back(match.queryIdx);
        train_idx.push_back(match.trainIdx);
        distance.push_back(match.distance);
    }

    vio_msgs::msg::DMatches ros_matches;
    ros_matches.img_idx.data = img_idx;
    ros_matches.query_idx.data = query_idx;
    ros_matches.train_idx.data = train_idx;
    ros_matches.distance.data = distance;

    return ros_matches;
}

static std::vector<cv::DMatch> OpenCVConversions::toCvDMatches(const vio_msgs::msg::DMatches &matches)
{
    std::vector<cv::DMatch> cv_matches;
    for (size_t i = 0; i < matches.img_idx.data.size(); i++)
    {
        cv::DMatch match(
            matches.query_idx.data[i],
            matches.train_idx.data[i],
            matches.img_idx.data[i],
            matches.distance.data[i]
        );

        cv_matches.push_back(match);
    }

    return cv_matches;

}

static sensor_msgs::msg::Image OpenCVConversions::toRosImage(const cv::Mat &image, std::string encoding = "mono8")
{
    cv_bridge::CvImage cv_image(std_msgs::msg::Header(), encoding, image);
    return *(cv_image.toImageMsg());
}

static cv::Mat OpenCVConversions::toCvImage(const sensor_msgs::msg::Image &image, std::string encoding = "mono8")
{
    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image, encoding);
    return cv_image->image;
}

#endif // __OPENCV_CONVERSIONS_HPP__