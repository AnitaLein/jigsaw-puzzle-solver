#include <QSettings>
#include <QFile>
#include <QDir>
#include <QList>
#include <QPointF>
#include <QRectF>
#include <QDebug>
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>
#include "mainWindow.h"
#include "ui_mainWindow.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

MainWindow::MainWindow(QWidget *parent): QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);

    QSettings settings(QSettings::IniFormat, QSettings::UserScope, "deisele", "puzzler");
    //QString baseUrl = settings.value("connection/baseUrl").toString();

    cv::namedWindow("Main", cv::WINDOW_NORMAL);

    std::vector<PuzzlePiece> puzzlePieces = loadImages("eda2", true);
    std::cout << "Loaded " << puzzlePieces.size() << " puzzle pieces" << std::endl;

    /*Edge referenceEdge = puzzlePieces[0].edges[2];
    std::cout << "Reference edge:\nFile name: " << puzzlePieces[0].scanFileName << ", Scan offset: " << puzzlePieces[0].scanOffset << std::endl;

    for (const PuzzlePiece &puzzlePiece: puzzlePieces) {
        for (int i = 0; i < 4; i++) {
            double similarity = compareEdges(referenceEdge, puzzlePiece.edges[i]);
            if (similarity < 1000) {
                std::cout << "\nPossible match:" << std::endl;
                std::cout << "Similarity: " << similarity << ", File name: " << puzzlePiece.scanFileName << ", Scan offset: " << puzzlePiece.scanOffset << ", Edge index: " << i << std::endl;
            }
        }
    }*/

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();

    // match every edge to every other edge and store the results in a matrix
    std::vector<std::vector<double>> similarityMatrix;
    for (const PuzzlePiece &puzzlePieceA: puzzlePieces) {
        for (int i = 0; i < 4; i++) {
            std::vector<double> similarities;

            for (const PuzzlePiece &puzzlePieceB: puzzlePieces) {
                for (int j = 0; j < 4; j++) {
                    double similarity;

                    // check edge continuity
                    if ((puzzlePieceA.edges[(i + 1) % 4].type == EdgeType::Flat) != (puzzlePieceB.edges[(j + 3) % 4].type == EdgeType::Flat) ||
                        (puzzlePieceA.edges[(i + 3) % 4].type == EdgeType::Flat) != (puzzlePieceB.edges[(j + 1) % 4].type == EdgeType::Flat)) {
                        similarity = std::numeric_limits<double>::infinity();
                    } else {
                        // ensure reflexivity
                        similarity = compareEdges(puzzlePieceA.edges[i], puzzlePieceB.edges[j]) + compareEdges(puzzlePieceB.edges[j], puzzlePieceA.edges[i]);
                    }

                    similarities.push_back(similarity);
                }
            }

            similarityMatrix.push_back(similarities);
        }

        std::cout << "." << std::flush;
    }

    std::cout << std::endl;

    auto t2 = high_resolution_clock::now();

    // getting number of milliseconds as an integer.
    //auto msInt = duration_cast<milliseconds>(t2 - t1);

    // getting number ofseconds as a double.
    duration<double> sDouble = t2 - t1;

    //std::cout << msInt.count() << "ms" << std::endl;
    std::cout << sDouble.count() << "s" << std::endl;

    /*
    // print similarity matrix
    for (const std::vector<double> &similarities: similarityMatrix) {
        for (double similarity: similarities) {
            std::cout << similarity << "\t";
        }
        std::cout << std::endl;
    }
    */

    // find lines with multiple matches with a similarity score below 500
    for (int i = 0; i < std::ssize(similarityMatrix); i++) {
        std::vector<int> hits;
        for (int j = 0; j < std::ssize(similarityMatrix[i]); j++) {
            if (similarityMatrix[i][j] < 600) {
                hits.push_back(j);
            }
        }

        if (hits.size() > 1) {
            const PuzzlePiece &puzzlePieceA = puzzlePieces[i / 4];
            std::cout << "\nPossible matches for edge " << i % 4 << " of " << puzzlePieceA.scanFileName << " at " << puzzlePieceA.scanOffset << ":" << std::endl;

            for (int hit: hits) {
                const PuzzlePiece &puzzlePieceB = puzzlePieces[hit / 4];
                std::cout << "Similarity: " << similarityMatrix[i][hit] << ", Edge " << hit % 4 << " of " << puzzlePieceB.scanFileName << " at " << puzzlePieceB.scanOffset << std::endl;
            }
        }
    }
}


MainWindow::~MainWindow() {
    delete ui;
}


std::vector<MainWindow::PuzzlePiece> MainWindow::loadImages(const std::string &directory, bool debugDraw) {
    std::vector<PuzzlePiece> puzzlePieces;

    QStringList files = QDir(directory.c_str()).entryList({"*b.jpg"}, QDir::Files, QDir::Name);

    for (const QString &filename: files) {
        std::string path = directory + "/" + filename.toStdString();

        cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        std::vector<PuzzlePiece> newPieces = extractPuzzlePieces(image, path);

        puzzlePieces.insert(puzzlePieces.end(), newPieces.begin(), newPieces.end());

        if (debugDraw) {
            cv::Mat foundPuzzlePieces;
            cv::cvtColor(image, foundPuzzlePieces, cv::COLOR_GRAY2BGR);

            // draw edges
            for (const PuzzlePiece &puzzlePiece: newPieces) {
                for (const Edge &edge: puzzlePiece.edges) {
                    cv::Scalar color;
                    switch (edge.type) {
                        case EdgeType::Flat:
                            color = cv::Scalar(0, 255, 255);
                            break;
                        case EdgeType::Tab:
                            color = cv::Scalar(0, 255, 0);
                            break;
                        case EdgeType::Blank:
                            color = cv::Scalar(0, 0, 255);
                            break;
                    }

                    for (int i = 0; i < std::ssize(edge.points) - 1; i++) {
                        cv::line(foundPuzzlePieces, edge.points[i] + cv::Point2d(puzzlePiece.scanOffset), edge.points[i + 1] + cv::Point2d(puzzlePiece.scanOffset), color, 2);
                    }
                }
            }

            // draw corners
            for (const PuzzlePiece &puzzlePiece: newPieces) {
                for (const Edge &edge: puzzlePiece.edges) {
                    cv::circle(foundPuzzlePieces, cv::Point2d(puzzlePiece.scanOffset) + edge.points[0], 5, cv::Scalar(255, 0, 0), cv::FILLED);
                }
            }

            std::string::size_type lastDot = path.find_last_of('.');
            cv::imwrite(path.substr(0, lastDot) + "_classified.png", foundPuzzlePieces);
            cv::imshow("Main", foundPuzzlePieces);
            cv::waitKey();
        }
    }

    return puzzlePieces;
}


std::vector<MainWindow::PuzzlePiece> MainWindow::extractPuzzlePieces(const cv::Mat &image, const std::string &filename) {
    std::vector<PuzzlePiece> puzzlePieces;

    std::vector<std::vector<cv::Point>> contours = extractContours(image, false);

    for (std::vector<cv::Point> contour: contours) {
        PuzzlePiece puzzlePiece;
        puzzlePiece.scanFileName = filename;

        cv::Rect boundingBox = cv::boundingRect(contour);
        puzzlePiece.scanOffset = boundingBox.tl();

        // move contour to origin
        for (cv::Point &p: contour) {
            p -= boundingBox.tl();
        }

        // extract region of interest
        cv::Mat roi = image(boundingBox);

        // create mask
        puzzlePiece.mask = cv::Mat::zeros(cv::boundingRect(contour).size(), CV_8UC1);
        cv::drawContours(puzzlePiece.mask, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), cv::FILLED);

        // cut out puzzle piece
        roi.copyTo(puzzlePiece.image, puzzlePiece.mask);

        BasicPuzzlePiece basicPuzzlePiece = fitBasicPuzzlePiece(contour);

        std::vector<cv::Point> puzzlePieceContour = createBasicPuzzlePiece(cv::boundingRect(contour), basicPuzzlePiece);
        refineContour(puzzlePiece.mask, puzzlePieceContour);

        // draw contour
        /*for (int i = 0; i < puzzlePieceContour.size(); i++) {
            cv::line(foundPuzzlePieces, puzzlePieceContour[i] + boundingBox.tl(), puzzlePieceContour[(i + 1) % puzzlePieceContour.size()] + boundingBox.tl(), cv::Scalar(255, 255, 0), 2);
        }*/

        // extract approximate corners from generated contour
        std::vector<cv::Point> corners;
        int i = 0;
        for (BasicEdge edge: basicPuzzlePiece.edges) {
            corners.push_back(puzzlePieceContour[i]);
            i += (edge.type == EdgeType::Flat? 1 : 5);
        }

        // slightly move corners diagonally outwards to improve accuracy
        std::vector<cv::Point> directions = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        for (int i = 0; i < 4; i++) {
            corners[i] += directions[i] * 5;
        }

        // find the indices of the nearest contour points
        std::vector<int> cornerIndices;
        for (cv::Point &corner: corners) {
            auto distToCorner = [&corner](const cv::Point &p) {
                return cv::norm(corner - p);
            };
            auto nearestCorner = std::ranges::min_element(contour, {}, distToCorner);

            cornerIndices.push_back(nearestCorner - contour.begin());
        }

        // split contour into edges
        for (int i = 0; i < 4; i++) {
            Edge edge;
            edge.type = basicPuzzlePiece.edges[i].type;

            for (int j = cornerIndices[i]; j != cornerIndices[(i + 1) % 4]; j = (j + 1) % contour.size()) {
                edge.points.push_back(contour[j]);
            }
            edge.points.push_back(contour[cornerIndices[(i + 1) % 4]]);

            puzzlePiece.edges.push_back(edge);
        }

        puzzlePieces.push_back(puzzlePiece);
    }

    return puzzlePieces;
}


std::vector<std::vector<cv::Point>> MainWindow::extractContours(const cv::Mat &image, bool b) {
    cv::Mat preprocessedImage;

    // median blur
    cv::Mat imageMedianBlur;
    cv::medianBlur(image, preprocessedImage, 5);

    // bilateral blur
    //cv::Mat imageBilateralBlur;
    //cv::bilateralFilter(image, imageBilateralBlur, 9, 75, 75);

    // otsu threshold
    cv::threshold(preprocessedImage, preprocessedImage, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

    // fill small holes in background and foreground
    if (b) {
        int kernelSize = 5;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
        cv::morphologyEx(preprocessedImage, preprocessedImage, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(preprocessedImage, preprocessedImage, cv::MORPH_CLOSE, kernel);
    }
    //cv::floodFill(image, cv::Point(0, 0), 125);

    // segment image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(preprocessedImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    std::erase_if(contours, [](const std::vector<cv::Point> &contour) {
        return cv::contourArea(contour) < 1000;
    });

    // find average contour height
    double averageHeight = 0;
    for (const std::vector<cv::Point> &contour: contours) {
        averageHeight += cv::boundingRect(contour).height;
    }
    averageHeight /= contours.size();

    // sort by contour center
    std::ranges::sort(contours, [width = image.cols, averageHeight](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
        auto gridOrder = [width, averageHeight](const std::vector<cv::Point> &contour) {
            cv::Moments moments = cv::moments(contour);
            cv::Point center(moments.m10 / moments.m00, moments.m01 / moments.m00);
            return center.y * width * 2 / averageHeight + center.x;
        };

        return gridOrder(a) < gridOrder(b);
    });

    // reverse all contours to get clockwise orientation
    for (std::vector<cv::Point> &contour: contours) {
        std::reverse(contour.begin(), contour.end());
    }

    return contours;
}


// contour must be at origin
MainWindow::BasicPuzzlePiece MainWindow::fitBasicPuzzlePiece(std::vector<cv::Point> contour) {
    cv::Mat mask = cv::Mat::zeros(cv::boundingRect(contour).size(), CV_8UC1);
    cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), cv::FILLED);

    BasicPuzzlePiece bestPuzzlePiece;
    int bestWhite = 10'000'000;

    EdgeType edgeTypes[] = {EdgeType::Flat, EdgeType::Tab, EdgeType::Blank};

    for (EdgeType top: edgeTypes) {
        for (EdgeType right: edgeTypes) {
            for (EdgeType bottom: edgeTypes) {
                for (EdgeType left: edgeTypes) {
                    for (double topOffset = -0.1; topOffset <= 0.1001; topOffset += 0.1) {
                        for (double rightOffset = -0.1; rightOffset <= 0.1001; rightOffset += 0.1) {
                            for (double bottomOffset = -0.1; bottomOffset <= 0.1001; bottomOffset += 0.1) {
                                for (double leftOffset = -0.1; leftOffset <= 0.1001; leftOffset += 0.1) {
                                    BasicPuzzlePiece curr{std::vector<BasicEdge>{{top, topOffset}, {right, rightOffset}, {bottom, bottomOffset}, {left, leftOffset}}};

                                    cv::Mat test = cv::Mat::zeros(cv::boundingRect(contour).size(), CV_8UC1);
                                    std::vector<cv::Point> puzzlePieceContour = createBasicPuzzlePiece(cv::boundingRect(contour), curr);
                                    cv::drawContours(test, std::vector<std::vector<cv::Point>>{puzzlePieceContour}, 0, cv::Scalar(255), cv::FILLED);

                                    cv::Mat difference;
                                    cv::absdiff(mask, test, difference);

                                    // count white pixels
                                    int white = cv::countNonZero(difference);
                                    if (white < bestWhite) {
                                        bestWhite = white;
                                        bestPuzzlePiece = curr;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return bestPuzzlePiece;
}


std::vector<cv::Point> MainWindow::createBasicPuzzlePiece(cv::Rect boundingBox, BasicPuzzlePiece piece) {
    double tabHeight = 0.3;
    double tabWidth = 0.25;
    double expansion = 0; //0.03;

    // construct a puzzle piece at with base length of 1, the tabs will protrude from the base
    QList<QPointF> points;

    points.append({0.0, 0.0});
    if (piece.edges[0].type != EdgeType::Flat) {
        points.append({(1 - tabWidth) / 2 + piece.edges[0].offset, 0.0});
        points.append({(1 - tabWidth) / 2 + piece.edges[0].offset - expansion, (piece.edges[0].type == EdgeType::Tab? -1 : 1) * tabHeight});
        points.append({(1 + tabWidth) / 2 + piece.edges[0].offset + expansion, (piece.edges[0].type == EdgeType::Tab? -1 : 1) * tabHeight});
        points.append({(1 + tabWidth) / 2 + piece.edges[0].offset, 0.0});
    }
    points.append({1.0, 0.0});

    if (piece.edges[1].type != EdgeType::Flat) {
        points.append({1.0, (1 - tabWidth) / 2 + piece.edges[1].offset});
        points.append({1 + (piece.edges[1].type == EdgeType::Tab? 1 : -1) * tabHeight, (1 - tabWidth) / 2 + piece.edges[1].offset - expansion});
        points.append({1 + (piece.edges[1].type == EdgeType::Tab? 1 : -1) * tabHeight, (1 + tabWidth) / 2 + piece.edges[1].offset + expansion});
        points.append({1.0, (1 + tabWidth) / 2 + piece.edges[1].offset});
    }
    points.append({1.0, 1.0});

    if (piece.edges[2].type != EdgeType::Flat) {
        points.append({(1 + tabWidth) / 2 + piece.edges[2].offset, 1.0});
        points.append({(1 + tabWidth) / 2 + piece.edges[2].offset + expansion, 1 + (piece.edges[2].type == EdgeType::Tab? 1 : -1) * tabHeight});
        points.append({(1 - tabWidth) / 2 + piece.edges[2].offset - expansion, 1 + (piece.edges[2].type == EdgeType::Tab? 1 : -1) * tabHeight});
        points.append({(1 - tabWidth) / 2 + piece.edges[2].offset, 1.0});
    }
    points.append({0.0, 1.0});

    if (piece.edges[3].type != EdgeType::Flat) {
        points.append({0.0, (1 + tabWidth) / 2 + piece.edges[3].offset});
        points.append({(piece.edges[3].type == EdgeType::Tab? -1 : 1) * tabHeight, (1 + tabWidth) / 2 + piece.edges[3].offset + expansion});
        points.append({(piece.edges[3].type == EdgeType::Tab? -1 : 1) * tabHeight, (1 - tabWidth) / 2 + piece.edges[3].offset - expansion});
        points.append({0.0, (1 - tabWidth) / 2 + piece.edges[3].offset});
    }
    //points.append({0.0, 0.0});

    // get the bounding box of the constructed puzzle piece
    QRectF rect = QRectF(0, 0, 1, 1);
    for (QPointF point: points) {
        rect = rect.united(QRectF(point, QPointF(0.5, 0.5)).normalized());
    }

    // normalize the puzzle piece to a unit square
    for (QPointF &point: points) {
        point.setX((point.x() - rect.x()) / rect.width());
        point.setY((point.y() - rect.y()) / rect.height());
    }

    // scale and translate the puzzle piece to the output bounding box
    std::vector<cv::Point> puzzlePiece;
    for (QPointF point: points) {
        puzzlePiece.push_back(cv::Point(std::round(boundingBox.x + (boundingBox.width - 1) * point.x()), std::round(boundingBox.y + (boundingBox.height - 1) * point.y())));
    }

    return puzzlePiece;
}


void MainWindow::refineContour(const cv::Mat &mask, std::vector<cv::Point> &contour, int n) {
    auto testContour = [&mask](std::vector<cv::Point> contour) {
        cv::Mat test = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::drawContours(test, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), cv::FILLED);

        cv::Mat difference;
        cv::absdiff(mask, test, difference);

        return cv::countNonZero(difference);
    };

    std::vector<cv::Point> directions = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}, {-1, -1}, {1, 1}, {-1, 1}, {1, -1}};
    int bestDiff = testContour(contour);

    for (int j = 0; j < n; j++) {
        bool changed = false;

        for (cv::Point &p: contour) {
            cv::Point bestDir = {0, 0};

            for (cv::Point dir: directions) {
                for (int factor = 1; factor <= 2; factor++) {
                    dir *= factor;
                    p += dir;
                    if (p.x >= 0 && p.x < mask.cols && p.y >= 0 && p.y < mask.rows) {
                        int diff = testContour(contour);
                        if (diff < bestDiff) {
                            bestDiff = diff;
                            bestDir = dir;

                            changed = true;
                        }
                    }

                    p -= dir;
                }
            }

            p += bestDir;
        }

        if (!changed) {
            break;
        }
    }
}


double MainWindow::compareEdges(const Edge &a, const Edge &b) {
    static int n = 0;

    if (a.type == b.type || a.type == EdgeType::Flat || b.type == EdgeType::Flat) {
        return std::numeric_limits<double>::infinity();
    }

    // transform b to match a (match corners)
    Transform transform = findTransformationLSQ({{a.points.front(), b.points.back()}, {a.points.back(), b.points.front()}});

    std::vector<std::pair<cv::Point2d, cv::Point2d>> matches;
    for (int i = 0; i < 10; i++) {
        // calculate the sum of squared distances
        matches = findMatchesClosest(a.points, b.points, transform);

        // improve the transformation by matching all points
        Transform newTransform = findTransformationLSQ(matches);
        if (newTransform == transform && i > 0) {
            //std::cout << "break" << std::endl;
            break;
        }

        transform = newTransform;

        if (i == 0) {
            double sum = 0;
            for (std::pair<cv::Point2d, cv::Point2d> match: matches) {
                double norm = cv::norm(match.first - transform(match.second));
                sum += norm * norm;
            }

            if (sum > 10'000) {
                return sum;
            }
        }
    }

    double sum = 0;
    for (std::pair<cv::Point2d, cv::Point2d> match: matches) {
        double norm = cv::norm(match.first - transform(match.second));
        sum += norm * norm;
    }

    if (sum > 2500 && sum < 3000) {
        // draw the two edges
        cv::Mat image = cv::Mat::zeros(cv::Size(600, 600), CV_8UC3);
        for (int i = 0; i < std::ssize(a.points) - 1; i++) {
            cv::line(image, a.points[i], a.points[i + 1], cv::Scalar(0, 255, 0), 1);
        }
        for (int i = 0; i < std::ssize(b.points) - 1; i++) {
            cv::line(image, transform(b.points[i]) + cv::Point2d(0, 0), transform(b.points[i + 1]) + cv::Point2d(0, 0), cv::Scalar(0, 0, 255), 1);
        }

        cv::imwrite("eda2/match_" + std::to_string(n) + ".png", image);

        //cv::imshow("edges", image);
        //cv::waitKey();

        // draw the two edges with offset
        image = cv::Mat::zeros(cv::Size(600, 600), CV_8UC3);
        for (int i = 0; i < std::ssize(a.points) - 1; i++) {
            cv::line(image, a.points[i], a.points[i + 1], cv::Scalar(0, 255, 0), 1);
        }
        for (int i = 0; i < std::ssize(b.points) - 1; i++) {
            cv::line(image, transform(b.points[i]) + cv::Point2d(0, 50), transform(b.points[i + 1]) + cv::Point2d(0, 50), cv::Scalar(0, 0, 255), 1);
        }

        cv::imwrite("eda2/match_offset_" + std::to_string(n) + ".png", image);

        //cv::imshow("edges", image);
        //cv::waitKey();

        n++;
    }

    return sum;
}


std::vector<std::pair<cv::Point2d, cv::Point2d>> MainWindow::findMatchesClosest(const std::vector<cv::Point2d> &a, const std::vector<cv::Point2d> &b, Transform transform) {
    std::vector<std::pair<cv::Point2d, cv::Point2d>> matches;
    for (cv::Point2d pointB: b) {
        cv::Point2d pointBTransformed = transform(pointB);
        auto itClosest = std::ranges::min_element(a, [pointBTransformed] (cv::Point2d a, cv::Point2d b) {
            return cv::norm(pointBTransformed - a) < cv::norm(pointBTransformed - b);
        });

        matches.push_back({*itClosest, pointB});
    }

    return matches;
}


MainWindow::Transform MainWindow::findTransformationLSQ(const std::vector<std::pair<cv::Point2d, cv::Point2d>> &matches) {
    Transform transform;

    if (matches.empty()) {
        return transform;
    }

    cv::Point2d aMean;
    cv::Point2d bMean;

    for (const std::pair<cv::Point2d, cv::Point2d> &match: matches) {
        aMean += match.first;
        bMean += match.second;
    }

    bMean /= (double)matches.size();
    aMean /= (double)matches.size();

    double s_xx = 0;
    double s_yy = 0;
    double s_xy = 0;
    double s_yx = 0;

    for (const std::pair<cv::Point2d, cv::Point2d> &match: matches) {
        s_xx += (match.second.x - bMean.x) * (match.first.x - aMean.x);
        s_yy += (match.second.y - bMean.y) * (match.first.y - aMean.y);

        s_xy += (match.second.x - bMean.x) * (match.first.y - aMean.y);
        s_yx += (match.second.y - bMean.y) * (match.first.x - aMean.x);
    }

    transform.w = std::atan2(s_xy - s_yx, s_xx + s_yy);
    //transform.t = aMean - bMean.rotated(transform.w);
    transform.t.x = aMean.x - (bMean.x * std::cos(transform.w) - bMean.y * std::sin(transform.w));
    transform.t.y = aMean.y - (bMean.x * std::sin(transform.w) + bMean.y * std::cos(transform.w));

    return transform;
}
