
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>


namespace fs = std::filesystem;
const std::string DATA_DIR = "../data";
const int SAMPLES_PER_LINE = 5;
const bool MATLAB_COORDS_ONE_BASED = true;

struct Line {
    cv::Point2d a, b;
    int vpId;
};

struct Camera {
    cv::Mat K, D;
};

struct SegmentRecord{
    cv::Point p1;
    cv::Point p2;
    int vpId;
};

Camera makeCamera(const double focal) {
    Camera cam;
    cam.K = (cv::Mat_<double>(3,3) <<
        focal, 0.0, 307.55130528,
        0.0, focal, 251.45424496,
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


std::vector<cv::Point2d> distortPoints(const std::vector<cv::Point2d>& pixels, const Camera& cam, double strength) {
    std::vector<cv::Point2d> normalized;
    normalized.reserve(pixels.size());

    double fx = cam.K.at<double>(0, 0);
    double fy = cam.K.at<double>(1, 1);
    double cx = cam.K.at<double>(0, 2);
    double cy = cam.K.at<double>(1, 2);

    for (const auto& p : pixels) {
        normalized.emplace_back((p.x - cx) / fx, (p.y - cy) / fy);
    }

    std::vector<cv::Point2d> fisheye;
    cv::fisheye::distortPoints(normalized, fisheye, cam.K, cam.D);
    std::vector<cv::Point2d> result;
    result.reserve(pixels.size());

    for (size_t i = 0; i < pixels.size(); ++i) {
        result.emplace_back(
            pixels[i].x * (1.0 - strength) +
            fisheye[i].x * strength,

            pixels[i].y * (1.0 - strength) +
            fisheye[i].y * strength
        );
    }

    return result;
}

std::unordered_map<int, cv::Point2d> loadTransformedVPs(const fs::path& inputPath, const Camera& cam, double strength){
    std::ifstream in(inputPath);
    if (!in) {
        throw std::runtime_error("Cannot open VP file: " + inputPath.string());
    }

    std::string row;
    std::getline(in, row); // Skip header

    const double fYork = 6.05317059 / 0.00896875;
    const double cxYork = 307.55130528;
    const double cyYork = 251.45424496;

    std::unordered_map<int, cv::Point2d> transformedVPs;

    int vpId = 0;

    while (std::getline(in, row)) {
        if (row.empty())
            continue;

        ++vpId;
        std::istringstream ss(row);
        double vx, vy, vz;
        if (!(ss >> vx >> vy >> vz)) 
            continue;
        // A direction with vz == 0 cannot be represented as a
        // finite point in this perspective projection.
        if (std::abs(vz) < 1e-10)  
            continue;

        cv::Point2d perspectiveVP(fYork * vx / vz + cxYork, fYork * vy / vz + cyYork);
        auto projected = distortPoints(std::vector<cv::Point2d>{perspectiveVP}, cam, strength);

        if (projected.empty())
            continue;

        transformedVPs.emplace(vpId, projected[0]);
    }

    return transformedVPs;
}

cv::Mat distortImage( const cv::Mat& src, const Camera& cam, double strength) {
    CV_Assert(strength >= 0.0);
    CV_Assert(strength <= 1.0);

    const int width  = src.cols;
    const int height = src.rows;

    std::vector<cv::Point2d> fisheyePixels;
    fisheyePixels.reserve(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            fisheyePixels.emplace_back(x, y);
        }
    }

    std::vector<cv::Point2d> sourcePixels;

    cv::fisheye::undistortPoints(fisheyePixels, sourcePixels, cam.K, cam.D, cv::Matx33d::eye(), cam.K);
    cv::Mat mapX(height, width, CV_32FC1);
    cv::Mat mapY(height, width, CV_32FC1);

    size_t idx = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++idx) {
            const double fisheyeX = sourcePixels[idx].x;
            const double fisheyeY = sourcePixels[idx].y;

            const double blendedX = (1.0 - strength) * x + strength * fisheyeX;
            const double blendedY = (1.0 - strength) * y + strength * fisheyeY;
            mapX.at<float>(y, x) = static_cast<float>(blendedX);
            mapY.at<float>(y, x) = static_cast<float>(blendedY);
        }
    }

    cv::Mat dst;
    cv::remap(src, dst, mapX, mapY, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return dst;
}

std::vector<Line> readLines(const fs::path& path) {
    std::vector<Line> lines;
    std::ifstream file(path);

    if (!file) { throw std::runtime_error("Cannot open lines file: " + path.string());}

    std::string row;
    std::getline(file, row);
    while (std::getline(file, row)) {
        std::istringstream ss(row);

        double x1, y1, x2, y2;
        int vpId;
        if (!(ss >> x1 >> y1 >> x2 >> y2 >> vpId)) {
            continue;
        }

        if (MATLAB_COORDS_ONE_BASED) {
            x1 -= 1.0; 
            y1 -= 1.0; 
            x2 -= 1.0; 
            y2 -= 1.0;
        }
        lines.push_back({{x1, y1}, {x2, y2}, vpId});
    }

    return lines;
}

void transformLines(const std::vector<Line>& lines, const Camera& cam, double strength, int width, int height, const fs::path& outputPath) {
    std::ofstream out(outputPath);
    if (!out) { 
        throw std::runtime_error( "Cannot write lines file: " + outputPath.string()); 
    }

    out << "x1 y1 x2 y2 vp_id\n";
    out << std::fixed << std::setprecision(6);

    constexpr double EXTENSION = 0;
    for (const auto& line : lines) {
        cv::Point2d dir = line.b - line.a;
        double len = cv::norm(dir);
        if (len < 1e-6)
            continue;

        dir /= len;
        cv::Point2d start = line.a - dir * EXTENSION;
        cv::Point2d end = line.b + dir * EXTENSION;
        std::vector<cv::Point2d> samples;
        samples.reserve(SAMPLES_PER_LINE);

        for (int i = 0; i < SAMPLES_PER_LINE; ++i) {
            double t = static_cast<double>(i) / (SAMPLES_PER_LINE - 1);
            samples.emplace_back(start * (1.0 - t) + end   * t);
        }

        auto curved = distortPoints(samples, cam, strength);
        for (size_t i = 1; i < curved.size(); ++i) {
            cv::Point p1(cvRound(curved[i - 1].x), cvRound(curved[i - 1].y));
            cv::Point p2(cvRound(curved[i].x), cvRound(curved[i].y));

            if (!cv::clipLine(cv::Size(width, height), p1, p2)) {
                continue;
            }

            out << p1.x << ' '
                << p1.y << ' '
                << p2.x << ' '
                << p2.y << ' '
                << line.vpId << '\n';
        }
    }
}



void transformLines_extended(const std::vector<Line>& lines, const Camera& cam, double strength, int width, int height, const fs::path& outputPath){
    std::ofstream out(outputPath);
    if (!out) {
        throw std::runtime_error("Cannot write lines file: " + outputPath.string());
    }

    out << "x1 y1 x2 y2 vp_id\n";
    out << std::fixed << std::setprecision(6);

    constexpr double EXTENSION = 1500;
    for (const auto& line : lines) {
        cv::Point2d dir = line.b - line.a;
        double len = cv::norm(dir);
        if (len < 1e-6)
            continue;

        dir /= len;
        cv::Point2d start = line.a - dir * EXTENSION;
        cv::Point2d end = line.b + dir * EXTENSION;

        std::vector<cv::Point2d> samples;
        samples.reserve(SAMPLES_PER_LINE);

        for (int i = 0; i < SAMPLES_PER_LINE; ++i) {
            double t = static_cast<double>(i) / (SAMPLES_PER_LINE - 1);
            samples.emplace_back(start * (1.0 - t) + end   * t);
        }

    
        auto curved = distortPoints(samples, cam, strength);
        for (size_t i = 1; i < curved.size(); ++i) {
            cv::Point p1(cvRound(curved[i - 1].x), cvRound(curved[i - 1].y));
            cv::Point p2(cvRound(curved[i].x), cvRound(curved[i].y));

            if (!cv::clipLine(cv::Size(width, height), p1, p2)){
                continue;
            }

            out << p1.x << ' '
                << p1.y << ' '
                << p2.x << ' '
                << p2.y << ' '
                << line.vpId << '\n';
        }
    }
}

void transformVPs(const fs::path& inputPath, const Camera& cam, double strength,const fs::path& outputPath) {
    std::ifstream in(inputPath);
    if (!in) {
        std::cerr << "Skipping missing VP file: " << inputPath << '\n';
        return;
    }

    std::ofstream out(outputPath);
    if (!out) {
        throw std::runtime_error("Cannot write VP file: " + outputPath.string());
    }

    out << "x y\n";
    out << std::fixed << std::setprecision(10);

    std::string row;
    std::getline(in, row);

    const double fYork = 6.05317059 / 0.00896875;
    const double cxYork = 307.55130528;
    const double cyYork = 251.45424496;

    while (std::getline(in, row)) {
        std::istringstream ss(row);
        double vx, vy, vz;

        if (!(ss >> vx >> vy >> vz)) {
            continue;
        }

        cv::Point2d vpPerspective(fYork * vx / vz + cxYork, fYork * vy / vz + cyYork);
        std::vector<cv::Point2d> pts{vpPerspective};
        auto distorted = distortPoints(pts, cam, strength);
        out << distorted[0].x << ' ' << distorted[0].y << '\n';
    }
}



void transformLines_splitByVPAlignment(const std::vector<Line>& lines, const Camera& cam,double strength, int width, int height, const std::unordered_map<int, cv::Point2d>& vps, const fs::path& towardOutputPath, const fs::path& awayOutputPath, const fs::path& randomOutputPath, double minCosine = 0.85){
    if (minCosine < 0.0 || minCosine > 1.0) {
        throw std::invalid_argument("minCosine must be between 0 and 1");
    }

    std::vector<SegmentRecord> towardSegments;
    std::vector<SegmentRecord> awaySegments;
    std::vector<SegmentRecord> allSegments;

    for (const auto& line : lines) {
        const auto vpIt = vps.find(line.vpId);

        if (vpIt == vps.end())
            continue;

        const cv::Point2d vp = vpIt->second;
        std::vector<cv::Point2d> samples;
        samples.reserve(SAMPLES_PER_LINE);

        for (int i = 0; i < SAMPLES_PER_LINE; ++i) {
            const double t = static_cast<double>(i) / (SAMPLES_PER_LINE - 1);
            samples.emplace_back(line.a * (1.0 - t) + line.b * t);
        }

        const auto curved = distortPoints(samples, cam, strength);

        for (size_t i = 1; i < curved.size(); ++i) {
            cv::Point2d tangent = curved[i] - curved[i - 1];
            double tangentLen = cv::norm(tangent);

            if (tangentLen < 1e-6)
                continue;

            tangent /= tangentLen;
            cv::Point2d midpoint = (curved[i] +  curved[i - 1]) * 0.5;
            cv::Point2d toVP = vp - midpoint;
            double vpDistance = cv::norm(toVP);

            if (vpDistance < 1e-6)
                continue;

            toVP /= vpDistance;
            const double alignment = std::abs(tangent.dot(toVP));
            cv::Point p1(cvRound(curved[i - 1].x),  cvRound(curved[i - 1].y));
            cv::Point p2(cvRound(curved[i].x), cvRound(curved[i].y));

            if (!cv::clipLine(cv::Size(width, height), p1, p2)) {
                continue;
            }

            SegmentRecord seg{p1, p2, line.vpId};
            allSegments.push_back(seg);

            if (alignment >= minCosine) {
                towardSegments.push_back(seg);
            }
            else {
                awaySegments.push_back(seg);
            }
        }
    }
    // Random control
    std::mt19937 rng(42);
    std::vector<SegmentRecord> randomSegments;
    std::shuffle(allSegments.begin(), allSegments.end(), rng);

    const size_t targetCount = towardSegments.size();
    randomSegments.insert(randomSegments.end(), allSegments.begin(), allSegments.begin() + std::min(targetCount,allSegments.size()));

    // Helper writer
    auto writeSegments = [](const fs::path& path, const std::vector<SegmentRecord>& segs) {
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("Cannot write: " + path.string());
        }

        out << "x1 y1 x2 y2 vp_id\n";
        for (const auto& s : segs) {
            out
                << s.p1.x << ' '
                << s.p1.y << ' '
                << s.p2.x << ' '
                << s.p2.y << ' '
                << s.vpId << '\n';
        }
    };

    writeSegments(towardOutputPath, towardSegments);
    writeSegments(awayOutputPath, awaySegments);
    writeSegments(randomOutputPath, randomSegments);
    /*
    std::cout
        << "Toward: "
        << towardSegments.size()
        << "\nAway: "
        << awaySegments.size()
        << "\nRandom: "
        << randomSegments.size()
        << "\nTotal: "
        << allSegments.size()
        << std::endl;
    
    std::cout
        << "Toward: " << towardSegments.size()
        << "\nAway: " << awaySegments.size()
        << "\nToward fraction: "
        << 100.0 * towardSegments.size() /
        allSegments.size()
        << "%\n";
        */
}


int main() {
    const fs::path root(DATA_DIR);
    const fs::path imageRoot = root / "YorkUrbanDB";
    const fs::path linesRoot = root / "lines";
    const fs::path vps3dRoot = root / "vps3d";

    
    std::vector<double> focals;

    for (double f = 675; f >= 100; f -= 25){
        focals.push_back(f);
    }
    
    for (const double focal : focals){
        std::cout << "\n=== FOCAL = "  << focal << " ===\n";

        const fs::path outputRoot = fs::path("../data/fisheye") / std::to_string(static_cast<int>(focal));
        
        const fs::path outImages = outputRoot / "images";
        const fs::path outLines = outputRoot / "lines";
        const fs::path outExtended = outputRoot / "extended_lines";
        const fs::path outVPs = outputRoot / "vps2d";
        const fs::path outToward= outputRoot / "lines_toward_vp";
        const fs::path outAway = outputRoot / "lines_away_vp";
        const fs::path outRandom = outputRoot / "lines_random_vp";

        fs::create_directories(outImages);
        fs::create_directories(outLines);
        fs::create_directories(outExtended);
        fs::create_directories(outVPs);
        fs::create_directories(outToward);
        fs::create_directories(outAway);
        fs::create_directories(outRandom);

        int processed = 0;
        int failed = 0;
        constexpr double DISTORTION = 1;
        for (const auto& entry : fs::directory_iterator(imageRoot)) {
            if (!entry.is_directory()) {
                continue;
            }

            const std::string id = entry.path().filename().string();
            const fs::path imagePath = entry.path() / (id + ".jpg");
            const fs::path linePath = linesRoot / (id + "_yud_vp.txt");
            const fs::path vpPath = vps3dRoot / (id + ".txt");

            try {cv::Mat src = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
                if (src.empty()) {
                    std::cerr << "Skipping missing image: " << imagePath << '\n';
                    ++failed;
                    continue;
                }

                Camera cam = makeCamera(focal);
                cv::Mat dst = distortImage(src, cam, DISTORTION);
                const fs::path outImagePath = outImages / (id + ".jpg");

                if (!cv::imwrite(outImagePath.string(), dst)) {
                    throw std::runtime_error("Failed to save image: " + outImagePath.string());
                }

                if (fs::exists(linePath)) {
                    auto lines = readLines(linePath);
                    transformLines(lines, cam, DISTORTION, src.cols, src.rows, outLines / (id + ".txt"));
                    transformLines_extended(lines, cam, DISTORTION, src.cols, src.rows, outExtended / (id + ".txt"));
                    auto vps = loadTransformedVPs(vpPath, cam, DISTORTION);

                    double minCosine = std::cos(5 * CV_PI / 180.0);
                    transformLines_splitByVPAlignment(lines, cam, DISTORTION, src.cols, src.rows, vps, outToward / (id + ".txt"), outAway / (id + ".txt"), outRandom / (id + ".txt"), minCosine);
                } else {
                    std::cerr << "Missing lines: " << linePath << '\n';
                }

                if (fs::exists(vpPath)) {
                    transformVPs(vpPath, cam, DISTORTION, outVPs / (id + ".txt")
                    );
                } else {
                    std::cerr << "Missing 3D VPs: " << vpPath << '\n';
                }

                ++processed;
                std::cout << "[" << processed << "] " << id << " processed\n";
            }
            catch (const std::exception& e) {
                ++failed;
                std::cerr << "Error processing " << id << ": " << e.what() << '\n';
            }
        }
    
      
    }
    return 0;
}

   
