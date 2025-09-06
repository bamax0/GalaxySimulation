#pragma once
#include "Physics/Star.hpp"
#include "ExportDLL.hpp"
#include <vector>

namespace cv { //Forward declaration
	class VideoWriter;
	class Mat;
}
struct Pixel
{
	unsigned char r, g, b;
};
class GALAXY_SIM_DLL_EXPORT ProbabilityComputer
{
	size_t m_imgSize;
	double m_maxValue{ -1 };
public:
	ProbabilityComputer(size_t s) : m_imgSize(s) {};

	void fillImage(cv::Mat* image, const std::vector<Star>& starVec, const double& posMax2);

private:
	size_t getMatrixCoord(const PosLy& s, const double& posMax2) const;
	Pixel getPixel(double coef) const;
};