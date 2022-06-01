#include <iostream>
#include <fstream>

#include <chrono>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

using Points3d = std::vector<cv::Point3f>;
using Points2d = std::vector<cv::Point2f>;

// def generate_pattern_points4d(size):
//     pattern_points = np.zeros((np.prod(size), 3), np.float64)
//     pattern_points[:, :2] = np.indices(size).T.reshape(-1, 2)
//     pattern_points *= 1
//
//     objp = np.zeros((1, size[0]*size[1], 3), np.float32)
//     objp[0, :, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
//
//     return objp
//
//

struct CalibrationResult {
    double rms{};
    cv::Mat K;
    cv::Mat D;
    cv::Mat stdDev;
    long time{};
};

void printResult(const CalibrationResult& result)
{
    std::cout << "rms time fx fy cx cy" << std::endl;
    std::cout << "D0..Dn" << std::endl;
    std::cout << "stdErr0..stdErrn" << std::endl;

    std::cout << result.rms << " ";
    std::cout << result.time << " ";
    std::cout << result.K.at<double>(0, 0) << " ";
    std::cout << result.K.at<double>(1, 1) << " ";
    std::cout << result.K.at<double>(0, 2) << " ";
    std::cout << result.K.at<double>(1, 2) << " ";
    std::cout << std::endl;
    std::cout << result.D << std::endl;
    std::cout << result.stdDev << std::endl;
}

cv::Size getImageSize(const std::string filepath)
{
    const cv::Mat image = cv::imread(filepath);
    if (image.empty())
    {
        return {0, 0};
    }
    else
    {
        return image.size();
    }
}

std::vector<std::vector<cv::Point2f>> detect_corners(const std::vector<std::string>& image_list,
                                                     const cv::Size patternSize = {6, 8})
{
    //    corners_set = []
    std::vector<std::vector<cv::Point2f>> cornersList;
    for (const auto& path : image_list)
    {
        const cv::Mat image = cv::imread(path, 0);
        if (image.empty())
        {
            std::cout << "could not read image from " << path << std::endl;
            continue;
        }
        std::vector<cv::Point2f> corners;
        std::vector<cv::Point2f> corners64;
        bool isFound = cv::findChessboardCorners(image, patternSize, corners);
        if (isFound)
        {
            // cv::TermCriteria criteria(CV_TERMCRIT_ITER + CV_TERMCRIT_ITER, 300, 0.01);
            cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 300, 0.01);
            cv::cornerSubPix(image, corners, {5, 5}, {-1, -1}, criteria);
            for (auto corner : corners)
            {
                corners64.emplace_back(corner.x, corner.y);
            }
            cornersList.push_back(corners64);
        }
        // cv::Mat debugImage = image.clone();
        // cv::drawChessboardCorners(debugImage, patternSize, corners, isFound);
        // cv::imshow("debug", debugImage);
        // cv::waitKey(0);
    }

    return cornersList;
}

std::vector<cv::Point3f> generatePatternPoints(const cv::Size boardSize, const double squareSize)
{
    std::vector<cv::Point3f> corners;

    for (int i = 0; i < boardSize.height; i++)
    {
        for (int j = 0; j < boardSize.width; j++)
        {
            corners.emplace_back(j * squareSize, i * squareSize, 0);
        }
    }

    return corners;
}

CalibrationResult calibrate_pinhole(const std::vector<Points3d>& points3d, const std::vector<Points2d>& points2d,
                       const cv::Size imageSize)
{
    // criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 300, 0.1) rms, K, D, _,
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 300, 1e-6);

    int flags = cv::CALIB_RATIONAL_MODEL;
    //flags |= cv::CALIB_THIN_PRISM_MODEL;
    //flags |= cv::CALIB_TILTED_MODEL;
    cv::Mat K, D, stdDev;
    const auto t1 = std::chrono::steady_clock::now();
    double rms = cv::calibrateCamera(points3d, points2d, imageSize, K, D, cv::noArray(),
                                     cv::noArray(), stdDev, cv::noArray(), cv::noArray(), flags, criteria);
    const auto t2 = std::chrono::steady_clock::now();

    CalibrationResult result;
    result.K = K;
    result.D = D;
    result.stdDev = stdDev.t();
    result.rms = rms;
    result.time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    return result;
}

CalibrationResult calibrate_fisheye(const std::vector<std::vector<cv::Point3f>>& objectPoints,
                       const std::vector<std::vector<cv::Point2f>>& cornersList, const cv::Size imageSize)
{
    // cv::TermCriteria criteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 3000, 0.1);
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 300, 1e-6);
    cv::Mat K, D;
    cv::Mat rvec, tvec;
    const auto t1 = std::chrono::steady_clock::now();
    double rms = cv::fisheye::calibrate(
        objectPoints, cornersList, imageSize, K, D, rvec, tvec,
        cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC + cv::fisheye::CALIB_CHECK_COND, criteria);
    const auto t2 = std::chrono::steady_clock::now();


    CalibrationResult result;
    result.K = K;
    result.D = D.t();
    result.rms = rms;
    result.time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    return result;
}

// def calibrate_omnidir(points3d, points2d, size):
//     t1 = timer()
//
//     K = np.empty((3, 3))
//     xi = None  # np.empty((0))
//     D = None  # np.empty((0))
//
//     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
//     rms, K, xi, D, _, _, _ = cv.omnidir.calibrate(
//         points3d, points2d, size, K, xi, D, 0, criteria)
//     return rms, K, D, timer() - t1
//
//
// def print_calibration(caption, rms, K, D, t):
//     print()
//     print(caption)
//     print(f'rms: {rms}')
//     print(f'K: {K}')
//     print(f'D: {D}')
//     print(f'time: {t * 1000} ms')
//
//
int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "usage: " << argv[0] << " image_list" << std::endl;
        return EXIT_FAILURE;
    }

    for (size_t i = 1; i < 4; ++i)
    {
        std::cout << "Dataset: " << argv[i] << std::endl;

        std::ifstream file(argv[i]);
        if (!file.is_open())
        {
            std::cout << "could not open " << argv[1] << std::endl;
            return EXIT_FAILURE;
        }

        std::vector<std::string> imageList;
        std::string line;
        while (std::getline(file, line))
        {
            imageList.push_back(line);
        }

        if (imageList.empty())
        {
            std::cout << "image list is empty" << std::endl;
            return EXIT_FAILURE;
        }

        const cv::Size pattern_size(6, 8);
        const cv::Size image_size = getImageSize(imageList[0]);

        const auto cornersList = detect_corners(imageList);

        std::vector<std::vector<cv::Point3f>> patternCornersList(
            cornersList.size(), generatePatternPoints(pattern_size, 1));

        //    rms, K, D, t = calibrate_pinhole(points3d, points2d, image_size)
        //    print_calibration('pinhole', rms, K, D, t)
        //
        printResult(calibrate_pinhole(patternCornersList, cornersList, image_size));
        printResult(calibrate_fisheye(patternCornersList, cornersList, image_size));
        //    print_calibration('fisheye', rms, K, D, t)
        //
        //    rms, K, D, t = calibrate_omnidir(points3d, points2d, image_size)
        //    print_calibration('omnidir', rms, K, D, t)
    }
}