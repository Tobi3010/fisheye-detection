
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct LineSegment {
    cv::Point2d p1, p2;
    int vp_id;
};

struct VanishingPoint {
    cv::Point2d p;
    int vp_id;
};


cv::Scalar vp_color(int vp_id) {
    switch (vp_id) {
        case 1: return cv::Scalar(0, 0, 255);     // Red
        case 2: return cv::Scalar(0, 255, 0);     // Green
        case 3: return cv::Scalar(255, 0, 0);     // Blue
        default: return cv::Scalar(128, 128, 128); // Gray
    }
}


std::vector<LineSegment> load_lines(const std::string& filename) {
    std::vector<LineSegment> lines;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open: " << filename << '\n';
        return lines;
    }

    std::string row;
    std::getline(file, row); // Skip header

    while (std::getline(file, row)) {
        if (row.empty()) continue;

        std::istringstream ss(row);
        double x1, y1, x2, y2;
        int vp_id;

        if (ss >> x1 >> y1 >> x2 >> y2 >> vp_id) {
            lines.push_back({
                {x1, y1}, {x2, y2}, vp_id
            });
        }
    }
    return lines;
}

std::vector<VanishingPoint> load_vps(
    const std::string& filename
) {
    std::vector<VanishingPoint> vps;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open: " << filename << '\n';
        return vps;
    }

    std::string row;
    std::getline(file, row); // Skip header

    while (std::getline(file, row)) {
        if (row.empty()) continue;

        std::istringstream ss(row);
        double x, y;
        
         int vp_id = static_cast<int>(vps.size()) + 1;

        if (ss >> x >> y) {
            vps.push_back({{x, y}, vp_id});
        }
    }
    return vps;
}



void draw_fisheye_lines(
    const std::vector<LineSegment>& lines,
    cv::Mat& img,
    cv::Point offset
) {
    for (size_t i = 0; i < lines.size(); i += 1) {
        const auto& line = lines[i];
        
        if (line.vp_id < 1 || line.vp_id > 3)
            continue;

        cv::Point p1(
            cvRound(line.p1.x + offset.x),
            cvRound(line.p1.y + offset.y)
        );
        cv::Point p2(
            cvRound(line.p2.x + offset.x),
            cvRound(line.p2.y + offset.y)
        );

        // Skip segments completely outside the canvas.
        if (!cv::clipLine(img.size(), p1, p2))
            continue;

        cv::line(
            img, p1, p2, vp_color(line.vp_id),
            2, cv::LINE_AA
        );
    }
}




void draw_fisheye_extended_lines(
    const std::vector<LineSegment>& lines,
    cv::Mat& img,
    cv::Point offset
) {
    const double dash_length  = 3.0;
    const double space_length = 9.0;
    const double step = dash_length + space_length;

    for (const auto& line : lines) {
        if (line.vp_id < 1 || line.vp_id > 3) continue;

        cv::Point p1(cvRound(line.p1.x + offset.x), cvRound(line.p1.y + offset.y));
        cv::Point p2(cvRound(line.p2.x + offset.x), cvRound(line.p2.y + offset.y));
        if (!cv::clipLine(img.size(), p1, p2)) continue;

        cv::Point2d d = p2 - p1;

        double length = cv::norm(d);

        if (length < 1e-6) continue;

        for (double s = 0.0; s < length; s += step) {
            double e = std::min(s + dash_length, length);

            double t0 = s / length;
            double t1 = e / length;

            cv::Point a(cvRound(p1.x + d.x * t0), cvRound(p1.y + d.y * t0));
            cv::Point b(cvRound(p1.x + d.x * t1), cvRound(p1.y + d.y * t1));
            cv::line(img, a, b, vp_color(line.vp_id), 1, cv::LINE_AA);
        }
    }
}

void draw_vps(
    const std::vector<VanishingPoint>& vps,
    cv::Mat& img,
    cv::Point offset
) {
    for (size_t i = 0; i < vps.size(); ++i) {
        cv::Point p(
            cvRound(vps[i].p.x + offset.x),
            cvRound(vps[i].p.y + offset.y)
        );

        if (p.x < 0 || p.x >= img.cols ||
            p.y < 0 || p.y >= img.rows)
            continue;

        cv::circle(
            img, p, 8, vp_color(vps[i].vp_id),
            -1, cv::LINE_AA
        );

    }
}

int main() {
    const std::string image_dir = "../data/fisheye/100/images/";
    const std::string lines_dir = "../data/fisheye/100/lines/";
    const std::string vps_dir   = "../data/fisheye/100/vps2d/";
     const std::string extended_dir   = "../data/fisheye/100/extended_lines/";

    for (const auto& entry :
         fs::directory_iterator(image_dir)) {

        if (!entry.is_regular_file())
            continue;

        std::string img_id = entry.path().stem().string();

        cv::Mat img = cv::imread(
            entry.path().string(), cv::IMREAD_COLOR
        );

        if (img.empty()) {
            std::cerr << "Could not load " << entry.path() << '\n';
            continue;
        }

        auto lines = load_lines(
            lines_dir + img_id + ".txt"
        );
        auto extended_lines = load_lines(
            extended_dir + img_id + ".txt"
        );
        auto vps = load_vps(
            vps_dir + img_id + ".txt"
        );

        // Extra room around the image.
        const int margin = 250;
        cv::Mat canvas(
            img.rows + 2 * margin,
            img.cols + 2 * margin,
            img.type(),
            cv::Scalar(0, 0, 0)
        );

        cv::Point offset(margin, margin);

        img.copyTo(
            canvas(cv::Rect(
                offset.x, offset.y, img.cols, img.rows
            ))
        );

        // Draw sampled, already-distorted line segments.
        draw_fisheye_lines(lines, canvas, offset);

        draw_fisheye_extended_lines(extended_lines, canvas, offset);

        // VP coordinates must already be fisheye-projected.
        draw_vps(vps, canvas, offset);

        cv::imshow("Fisheye: " + img_id, canvas);

        std::cout << img_id
                  << " | lines: " << lines.size()
                  << " | VPs: " << vps.size()
                  << "\nPress any key for the next image.\n";

        if (cv::waitKey(0) == 27) // Esc to quit
            break;
    }

    cv::destroyAllWindows();
    return 0;
}
