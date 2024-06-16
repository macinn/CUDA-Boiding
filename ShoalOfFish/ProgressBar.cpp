#include <iostream>
#include <string>

class ProgressBar
{
    int barWidth;
    int colorCode;
    float progress;
    std::string label;
public:
    ProgressBar(std::string label = "", int barWidth = 70, float progress = 0.0, int colorCode = 32) : barWidth(barWidth), progress(progress), colorCode(colorCode), label(label) {
        if(label.length() > 0)
            this->barWidth = barWidth - label.length() - 1;
    }

    float getProgress() const {
        return progress;
    }
    void setProgress(float progress) {
        this->progress = progress;
    }
    void incrementProgress(float increment) {
        this->progress += increment;
    }
    void display() const {
        std::cout << "\r" << label << " [" << "\x1B[" << colorCode << "m";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "\033[0m" << "] " << int(progress * 100.0) << "% ";
        std::cout.flush();
    }
    std::string getLabel() const {
		return label;
	}
};