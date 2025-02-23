#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QMainWindow>
#include <vector>
#include <opencv2/core/core.hpp>

namespace Ui {
    class MainWindow;
}

class MainWindow: public QMainWindow {
        Q_OBJECT

    public:
        struct Transform {
            Transform(): w(0) {};
            Transform(cv::Point2d t, double w): t(t), w(w) {};

            friend bool operator==(const Transform &a, const Transform &b) = default;

            cv::Point2d operator()(const cv::Point2d &p) const {
                return {p.x * std::cos(w) - p.y * std::sin(w) + t.x,
                        p.x * std::sin(w) + p.y * std::cos(w) + t.y};
            }

            cv::Point2d t;
            double w;
        };

        enum class EdgeType {
            Flat,
            Tab,
            Blank
        };

        struct BasicEdge {
            EdgeType type;
            double offset;
        };

        struct BasicPuzzlePiece {
            std::vector<BasicEdge> edges;
        };

        struct Edge {
            EdgeType type;
            std::vector<cv::Point2d> points;
        };

        struct PuzzlePiece {
            cv::Mat image;
            cv::Mat mask;
            std::vector<Edge> edges;

            std::string scanFileName;
            cv::Point scanOffset;
        };

        MainWindow(QWidget *parent = nullptr);
        ~MainWindow();

        std::vector<PuzzlePiece> loadImages(const std::string &directory, bool debugDraw = false);
        std::vector<PuzzlePiece> extractPuzzlePieces(const cv::Mat &image, const std::string &filename);

        std::vector<std::vector<cv::Point>> extractContours(const cv::Mat &image, bool b = false);

        BasicPuzzlePiece fitBasicPuzzlePiece(std::vector<cv::Point> contour);
        std::vector<cv::Point> createBasicPuzzlePiece(cv::Rect boundingBox, BasicPuzzlePiece piece);

        void refineContour(const cv::Mat &mask, std::vector<cv::Point> &contour, int n = 100);

        double compareEdges(const Edge &a, const Edge &b);
        std::vector<std::pair<cv::Point2d, cv::Point2d>> findMatchesClosest(const std::vector<cv::Point2d> &a, const std::vector<cv::Point2d> &b, Transform transform);
        Transform findTransformationLSQ(const std::vector<std::pair<cv::Point2d, cv::Point2d>> &matches);

    private:
        Ui::MainWindow *ui;
};

#endif
