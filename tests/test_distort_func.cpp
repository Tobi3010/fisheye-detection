#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

struct Camera {
    cv::Mat K, D;
};

Camera makeCamera()
{
    Camera cam;

    cam.K = (cv::Mat_<double>(3,3) <<
        150.0, 0.0, 307.55130528,
        0.0, 150.0, 251.45424496,
        0.0, 0.0, 1.0
    );

    cam.D = (cv::Mat_<double>(4,1) <<
        0.02,
        0.002,
        0.0,
        0.0
    );

    return cam;
}

cv::Mat distortImage(
    const cv::Mat& src,
    const Camera& cam,
    double strength)   // 0.0 .. 1.0
{
    CV_Assert(strength >= 0.0);
    CV_Assert(strength <= 1.0);

    const int width  = src.cols;
    const int height = src.rows;

    std::vector<cv::Point2d> fisheyePixels;
    fisheyePixels.reserve(width * height);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            fisheyePixels.emplace_back(x, y);
        }
    }

    std::vector<cv::Point2d> sourcePixels;

    cv::fisheye::undistortPoints(
        fisheyePixels,
        sourcePixels,
        cam.K,
        cam.D,
        cv::Matx33d::eye(),
        cam.K
    );

    cv::Mat mapX(height, width, CV_32FC1);
    cv::Mat mapY(height, width, CV_32FC1);

    size_t idx = 0;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x, ++idx)
        {
            const double fisheyeX = sourcePixels[idx].x;
            const double fisheyeY = sourcePixels[idx].y;

            const double blendedX =
                (1.0 - strength) * x +
                strength * fisheyeX;

            const double blendedY =
                (1.0 - strength) * y +
                strength * fisheyeY;

            mapX.at<float>(y, x) =
                static_cast<float>(blendedX);

            mapY.at<float>(y, x) =
                static_cast<float>(blendedY);
        }
    }

    cv::Mat dst;

    cv::remap(
        src,
        dst,
        mapX,
        mapY,
        cv::INTER_CUBIC,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0)
    );

    return dst;
}

int main()
{
    cv::Mat src = cv::imread("../data/YorkUrbanDB/P1020861/P1020861.jpg");

    if (src.empty())
    {
        std::cerr << "Could not load image\n";
        return 1;
    }

    Camera cam = makeCamera();

    cv::Mat dst = distortImage(src, cam, 1);

    cv::Mat diff;
    cv::absdiff(src, dst, diff);

    double minVal, maxVal;
    cv::minMaxLoc(diff, &minVal, &maxVal);

    double l1 = cv::norm(src, dst, cv::NORM_L1);
    double l2 = cv::norm(src, dst, cv::NORM_L2);

    cv::Scalar meanDiff = cv::mean(diff);

    std::cout << "\n===== IMAGE DIFFERENCE TEST =====\n";
    std::cout << "Max pixel difference : " << maxVal << "\n";
    std::cout << "L1 difference        : " << l1 << "\n";
    std::cout << "L2 difference        : " << l2 << "\n";
    std::cout << "Mean difference BGR  : "
              << meanDiff[0] << ", "
              << meanDiff[1] << ", "
              << meanDiff[2] << "\n";

    cv::imwrite("original.png", src);
    cv::imwrite("transformed.png", dst);

    cv::Mat vis;
    diff.convertTo(vis, CV_8U);

    cv::imwrite("difference.png", vis);

    return 0;
}