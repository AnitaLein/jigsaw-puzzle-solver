#include <QApplication>
#include <QTranslator>
#include <QLocale>
#include "mainWindow.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);

    QApplication::setOrganizationName("UT");
    QApplication::setApplicationName("Jigsaw Puzzle Solver");
    QApplication::setApplicationVersion("0.0.1");
    //bQApplication::setAttribute(Qt::AA_DisableWindowContextHelpButton);

    QTranslator defaultQtTranslator;
    (void)defaultQtTranslator.load(QLocale(), "qtbase", "_", ":/i18n");
    QApplication::installTranslator(&defaultQtTranslator);

    /*QTranslator produktkatalogTranslator;
    produktkatalogTranslator.load(QLocale(), "produktkatalog", "_", ":/i18n");
    QApplication::installTranslator(&produktkatalogTranslator);*/

    MainWindow w;
    w.show();
    return a.exec();
}
