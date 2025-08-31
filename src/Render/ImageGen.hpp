#pragma once

#include "Physics/Star.hpp"
#include "ExportDLL.hpp"

#include <filesystem>
#include <iostream>
#include <stdio.h>

namespace cv {
	class VideoWriter;
	class Mat;
}

// Fonction definie dans le cpp
// cv::Vec3b colorTemperatureToRGB(double kelvin);
// double getTemperatureFromMas(double m);
// unsigned char clampToUchar(double x, double min, double max);

class GALAXY_SIM_DLL_EXPORT ImageGen
{
	static const size_t img_size = 2048;
	std::filesystem::path m_path;
public:

	void reinit();
	void init(const std::filesystem::path& parentFolder);

	void drawPixel(cv::Mat* val, const double& xr, const double& yr, double mass) const;

	void generateImg(const std::vector<Star>& starVec, const dVec3& camPos, const double& posMax2, size_t i);

	cv::VideoWriter* m_cap{ nullptr };
	// cv::VideoWriter* cap2;
};