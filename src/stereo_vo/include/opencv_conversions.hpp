#ifndef __OPENCV_CONVERSIONS_HPP__
#define __OPENCV_CONVERSIONS_HPP__

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include "vio_msgs/msg/d_matches.hpp"
#include "vio_msgs/msg/key_points.hpp"
#include "std_msgs/msg/header.hpp"

namespace OpenCVConversions
{

    vio_msgs::msg::KeyPoints toRosKeyPoints(const std::vector<cv::KeyPoint> &keypoints)
    {
        std::vector<float> angle, response, size;
        std::vector<int> class_id, octave, pt_x, pt_y;

        vio_msgs::msg::KeyPoints ros_keypoints;
        for (const cv::KeyPoint &kp : keypoints)
        {
            ros_keypoints.angle.data.push_back(kp.angle);
            ros_keypoints.response.data.push_back(kp.response);
            ros_keypoints.size.data.push_back(kp.size);
            ros_keypoints.class_id.data.push_back(kp.class_id);
            ros_keypoints.octave.data.push_back(kp.octave);
            ros_keypoints.pt_x.data.push_back(kp.pt.x);
            ros_keypoints.pt_y.data.push_back(kp.pt.y);
        }

        return ros_keypoints;
    }

    std::vector<cv::KeyPoint> toCvKeyPoints(const vio_msgs::msg::KeyPoints &keypoints)
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

    vio_msgs::msg::DMatches toRosDMatches(const std::vector<cv::DMatch> &matches)
    {
        std::vector<int> img_idx, query_idx, train_idx;
        std::vector<float> distance;

        vio_msgs::msg::DMatches ros_matches;
        for (const cv::DMatch &match : matches)
        {
            ros_matches.img_idx.data.push_back(match.imgIdx);
            ros_matches.query_idx.data.push_back(match.queryIdx);
            ros_matches.train_idx.data.push_back(match.trainIdx);
            ros_matches.distances.data.push_back(match.distance);
        }

        return ros_matches;
    }

    std::vector<cv::DMatch> toCvDMatches(const vio_msgs::msg::DMatches &matches)
    {
        std::vector<cv::DMatch> cv_matches;
        for (size_t i = 0; i < matches.img_idx.data.size(); i++)
        {
            cv::DMatch match(
                matches.query_idx.data[i],
                matches.train_idx.data[i],
                matches.img_idx.data[i],
                matches.distances.data[i]
            );

            cv_matches.push_back(match);
        }

        return cv_matches;

    }

    sensor_msgs::msg::Image toRosImage(const cv::Mat &image, std::string encoding = "mono8")
    {
        cv_bridge::CvImage cv_image(std_msgs::msg::Header(), encoding, image);
        return *(cv_image.toImageMsg());
    }

    cv::Mat toCvImage(const sensor_msgs::msg::Image &image, std::string encoding = "mono8")
    {
        cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image, encoding);
        return cv_image->image;
    }

}

#endif // __OPENCV_CONVERSIONS_HPP__