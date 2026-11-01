#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


using namespace cv;
using namespace std;
namespace fs = std::filesystem;


struct LineSegment {
    cv::Point2d p1;
    cv::Point2d p2;
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
        std::cerr << "Could not open: " << filename << std::endl;
        return lines;
    }

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::stringstream ss(line);

        double x1, y1, x2, y2;
        int vp_id;

        ss >> x1 >> y1 >> x2 >> y2 >> vp_id;

        lines.push_back({cv::Point2d(x1, y1), cv::Point2d(x2, y2), vp_id});
    }
    return lines;
}





void draw_lines(const std::vector<LineSegment>& lines, cv::Mat& img, int offset_x, int offset_y) {
     for (const auto& line : lines) {
        if (line.vp_id < 1 || line.vp_id > 3){
            continue;
        }

        cv::Point p1(cvRound(line.p1.x + offset_x), cvRound(line.p1.y + offset_y));
        cv::Point p2(cvRound(line.p2.x + offset_x),cvRound(line.p2.y + offset_y));

        cv::line(img, p1, p2, vp_color(line.vp_id), 2, cv::LINE_AA);
    }
}


void draw_extended_lines(const std::vector<LineSegment>& lines, cv::Mat& img, int offset_x, int offset_y) {
    // Definer længden på stregen og mellemrummet i pixels
    const double dash_length = 10.0;
    const double space_length = 10.0;
    const double step = dash_length + space_length;

    for (const auto& line : lines) {
        if (line.vp_id < 1 || line.vp_id > 3){
            continue;
        }

        cv::Point2d direction = line.p2 - line.p1;
        double length = std::sqrt(direction.x * direction.x + direction.y * direction.y);

        if (length < 1e-6)
            continue;

        // Normalize
        direction /= length;
        double extension = 2000.0;

        cv::Point2d start = cv::Point2d(line.p1.x + offset_x, line.p1.y + offset_y) - direction * extension;
        cv::Point2d end = cv::Point2d(line.p2.x + offset_x, line.p2.y + offset_y) + direction * extension;

        // Beregn den fulde længde af den udvidede linje
        cv::Point2d extended_dir = end - start;
        double extended_length = std::sqrt(extended_dir.x * extended_dir.x + extended_dir.y * extended_dir.y);

        // Tegn stiplede linjesegmenter
        for (double i = 0; i < extended_length; i += step) {
            // Starten af det lille segment
            double r_start = i / extended_length;
            int x_start = cvRound(start.x * (1.0 - r_start) + end.x * r_start);
            int y_start = cvRound(start.y * (1.0 - r_start) + end.y * r_start);

            // Enden af det lille segment (må ikke gå ud over den maksimale længde)
            double current_dash_end = std::min(i + dash_length, extended_length);
            double r_end = current_dash_end / extended_length;
            int x_end = cvRound(start.x * (1.0 - r_end) + end.x * r_end);
            int y_end = cvRound(start.y * (1.0 - r_end) + end.y * r_end);

            // Tegn det lille segment
            cv::line(img,
                cv::Point(x_start, y_start),
                cv::Point(x_end, y_end),
                vp_color(line.vp_id),
                1,
                cv::LINE_AA
            );
        }
    }
}



std::vector<VanishingPoint> load_vps(const std::string& filename) {
    std::vector<VanishingPoint> vps;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open: " << filename << std::endl;
        return vps;
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::stringstream ss(line);

        double x, y;
        if (!(ss >> x >> y))
            continue;

        int vp_id = static_cast<int>(vps.size()) + 1;

        vps.push_back({
            cv::Point2d(x, y),
            vp_id
        });
    }

    return vps;
}


void draw_vps(std::vector<VanishingPoint>& vps, cv::Mat& img, int offset_x, int offset_y) {
   
    for (VanishingPoint& vp : vps) {
        cv::Point p(cvRound(vp.p.x + offset_x), cvRound(vp.p.y + offset_y));

        if (p.x < 0 || p.x >= img.cols ||  p.y < 0 || p.y >= img.rows) {
            continue;
        }

        cv::circle(img, p, 10, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
        cv::circle(img, p, 8, vp_color(vp.vp_id), -1, cv::LINE_AA);
    }
}





int main() {

    const std::string base_path = "../data/YorkUrbanDB/";
    const std::string lines_base = "../data/lines/";
    const std::string vps2d_base = "../data/vps2d/";

    // Iterate through every folder in YorkUrbanDB
    for (const auto& entry : fs::directory_iterator(base_path)) {

        if (!entry.is_directory())
            continue;

        // Folder name is the image ID, e.g. P1020174
        std::string img_id = entry.path().filename().string();

        std::string yud_path =
            base_path + img_id + "/" + img_id + ".jpg";

        std::string lines_path =
            lines_base + img_id + "_yud_vp.txt";

        std::string vps2d_path =
            vps2d_base + img_id + ".txt";

        cv::Mat img = cv::imread(yud_path);

        // Skip if the image cannot be loaded
        if (img.empty()) {
            std::cerr << "Could not load image: "
                      << yud_path << std::endl;
            continue;
        }

        // Create white canvas
        const int canvas_w = 1400;
        const int canvas_h = 1000;

        cv::Mat canvas(
            canvas_h, canvas_w, img.type(),
            cv::Scalar(255, 255, 255)
        );

        int offset_x = (canvas_w - img.cols) / 2;
        int offset_y = (canvas_h - img.rows) / 2;

        // Copy image onto canvas
        img.copyTo(
            canvas(cv::Rect(
                offset_x, offset_y, img.cols, img.rows
            ))
        );

        // Load and draw lines
        std::vector<LineSegment> lines =
            load_lines(lines_path);

        draw_extended_lines(lines, canvas, offset_x, offset_y);
        draw_lines(lines, canvas, offset_x, offset_y);

        // Load and draw vanishing points
        std::vector<VanishingPoint> vps =
            load_vps(vps2d_path);

        draw_vps(vps, canvas, offset_x, offset_y);

        // Display current image
        cv::imshow("Displayed Image - " + img_id, canvas);

        std::cout << "Displaying: " << img_id << std::endl;

        // Press any key to move to the next image
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return 0;
}
