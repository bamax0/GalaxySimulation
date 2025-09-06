#pragma once

#include "Physics/Star.hpp"
#include "ExportDLL.hpp"

#include <filesystem>
#include <iostream>
#include <stdio.h>
#include "ProbabilityComputer.hpp"

namespace cv { //Forward declaration
	class VideoWriter;
	class Mat;
}

// Fonction definie dans le cpp
// cv::Vec3b colorTemperatureToRGB(double kelvin);
// double getTemperatureFromMas(double m);
// unsigned char clampToUchar(double x, double min, double max);

static const size_t k_imgSize = 2048;
class GALAXY_SIM_DLL_EXPORT ImageGen
{
	std::filesystem::path m_path;
	ProbabilityComputer m_computer{ k_imgSize };
public:

	void reinit();
	void init(const std::filesystem::path& parentFolder);


	size_t convertToImageCoordinate(const double& x);
	void drawPixel(cv::Mat* val, const double& xr, const double& yr, double mass) const;

	void generateImg(const std::vector<Star>& starVec, const dVec3& camPos, const double& posMax2, size_t i);

	cv::VideoWriter* m_cap{ nullptr };
	// cv::VideoWriter* cap2;
};