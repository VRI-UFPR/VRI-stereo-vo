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
    class CameraIntrinsics
    {
    private:
        cv::Mat camera_matrix, dist_coeffs, rot_matrix, tl_vector;
    public:
        CameraIntrinsics() {}

        CameraIntrinsics(const std::string &intrinsics_file) 
        {
            cv::FileStorage fs(intrinsics_file, cv::FileStorage::READ);
            fs["K"] >> this->camera_matrix;
            fs["D"] >> this->dist_coeffs;

            cv::Mat body_T_cam;
            fs["body_T_cam"] >> body_T_cam;
            fs.release();

            this->rot_matrix = body_T_cam.rowRange(0, 3).colRange(0, 3);
            this->tl_vector = body_T_cam.rowRange(0, 3).col(3);
        }

        // Copy constructor
        CameraIntrinsics& operator=(const CameraIntrinsics &other)
        {
            this->camera_matrix = other.camera_matrix.clone();
            this->dist_coeffs = other.dist_coeffs.clone();
            this->rot_matrix = other.rot_matrix.clone();
            this->tl_vector = other.tl_vector.clone();

            return *this;
        }

        cv::Mat undistortImage(const cv::Mat &img)
        {
            cv::Mat undistorted_img;
            cv::undistort(img, undistorted_img, this->camera_matrix, this->dist_coeffs);

            return undistorted_img;
        }

        // Getters
        double fx() const { return this->camera_matrix.at<double>(0, 0); }
        double fy() const { return this->camera_matrix.at<double>(1, 1); }
        double cx() const { return this->camera_matrix.at<double>(0, 2); }
        double cy() const { return this->camera_matrix.at<double>(1, 2); }
        cv::Mat cameraMatrix() const { return this->camera_matrix; }
        cv::Mat distCoeffs() const { return this->dist_coeffs; }
        cv::Mat rotMatrix() const { return this->rot_matrix; }
        cv::Mat tlVector() const { return this->tl_vector; }
    };

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
    
    std::string cvEnc2Str(const int cv_encoding)
    {
        switch (cv_encoding)
        {
        case CV_8UC1:
            return "mono8";
        
        case CV_8UC3:
            return "bgr8";

        case CV_16UC1:
            return "mono16";

        case CV_16UC3:
            return "bgr16";

        case CV_32FC1:
            return "32FC1";

        default:
            return "mono8";
        }
    }

    sensor_msgs::msg::Image toRosImage(const cv::Mat &image)
    {
        cv_bridge::CvImage cv_image(std_msgs::msg::Header(), OpenCVConversions::cvEnc2Str(image.type()), image);
        return *(cv_image.toImageMsg());
    }

    cv::Mat toCvImage(const sensor_msgs::msg::Image &image)
    {
        cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image, image.encoding);
        return cv_image->image;
    }

}

#endif // __OPENCV_CONVERSIONS_HPP__