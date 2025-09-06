#include "Render/ImageGen.hpp"

// TODO
#include "../../external/opencv/build/include/opencv2/video.hpp"
#include "../../external/opencv/build/include/opencv2/videoio.hpp"
#include "../../external/opencv/build/include/opencv2/imgcodecs.hpp"
#include "../../external/opencv/build/include/opencv2/imgproc.hpp"
#include "../../external/opencv/build/include/opencv2/core/mat.hpp"
#include "../../external/opencv/build/include/opencv2/videoio/videoio.hpp"
#include "ProbabilityComputer.hpp"

static double getTemperatureFromMas(double m) // m en solar mass
{
	// m /= SOLAR_MASS * 1e7;
	return 5000; // 3000. + 1666. * m + 666. * m * m;
	// return m;
}

static unsigned char clampToUchar(double x, double min, double max)
{
	if (x < min) { return static_cast<unsigned char>(min); }
	if (x > max) { return static_cast<unsigned char>(max); }
	return static_cast<unsigned char>(x);
}

static cv::Vec3b colorTemperatureToRGB(double kelvin) {
	// https://gist.github.com/paulkaplan/5184275
	double temp = kelvin / 100.;
	double red, green, blue;

	if (temp <= 66) {
		red = 255;
		green = temp;
		green = 99.4708025861 * std::log(green) - 161.1195681661;

		if (temp <= 19) {
			blue = 0;
		}
		else {
			blue = temp - 10;
			blue = 138.5177312231 * std::log(blue) - 305.0447927307;
		}
	}
	else {
		red = temp - 60;
		red = 329.698727446 * std::pow(red, -0.1332047592);
		green = temp - 60;
		green = 288.1221695283 * std::pow(green, -0.0755148492);
		blue = 255;
	}
	return cv::Vec3b{ clampToUchar(red,   0, 255),  clampToUchar(green, 0, 255), clampToUchar(blue,  0, 255) };
}

void ImageGen::reinit()
{
	std::filesystem::path parentFolder{m_path.parent_path()};
	int i = 0;
	while (std::filesystem::exists(parentFolder / ("video" + std::to_string(i) + ".avi")))
	{
		++i;
	}
	std::string imgPath = (parentFolder / u8"video").string() + std::to_string(i) + ".avi"; // 'DIVX'

	if (m_cap != nullptr) delete m_cap;
	m_cap = new cv::VideoWriter();
	int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
	m_cap->open(imgPath, codec, 20, cv::Size{ k_imgSize , k_imgSize }, true);
	if (!m_cap->isOpened())
	{
		std::cout << "Error, video non ouverte" << std::endl;
	}

}

void ImageGen::init(const std::filesystem::path& parentFolder)
{
	m_path = parentFolder / u8"Images";
	if (!std::filesystem::exists(m_path))
	{
		std::filesystem::create_directories(m_path);
	}
	reinit();
	// cap2 = new cv::VideoWriter((p / "_octree" / "_video.avi").string(), 0, 30, { img_size, img_size });
}

void ImageGen::drawPixel(cv::Mat* val, const double& xr, const double& yr, double mass) const
{
	size_t x = size_t(xr) + k_imgSize / 2;
	size_t y = size_t(yr) + k_imgSize / 2;
	if (
		x < k_imgSize && x >= 0 &&
		y < k_imgSize && y >= 0
		)
	{
		auto c  = colorTemperatureToRGB(getTemperatureFromMas(mass));
		// std::cout << c << std::endl;
		val->at<cv::Vec3b>(int(x), int(y)) = c;
	}
}

void ImageGen::generateImg(const std::vector<Star>& starVec, const dVec3& camPos, const double& posMax2, size_t i)
{
	cv::Mat val = cv::Mat::zeros(k_imgSize, k_imgSize, CV_8UC3);

	m_computer.fillImage(&val, starVec, posMax2);
	// float rap = static_cast<float>(k_imgSize) / static_cast<float>(posMax2);
	// for (const Star& star : starVec)
	// {
	// 	auto newPos = star.position * rap;
	// 	drawPixel(&val, newPos.x, newPos.y, star.mass);
	// }
	cv::Mat blurred;
	// cv::GaussianBlur(val, blurred, { 3, 3 }, 2.5);

	size_t kernelSize{ 7u };
	cv::Mat kernel= cv::Mat::ones(kernelSize, kernelSize, CV_32F) / (float)(kernelSize * kernelSize);
	kernel.at<float>(0, 0) = 0.00; kernel.at<float>(1, 0) = 0.00; kernel.at<float>(2, 0) = 0.00; kernel.at<float>(3, 0) = 0.20; kernel.at<float>(4, 0) = 0.00; kernel.at<float>(5, 0) = 0.00; kernel.at<float>(6, 0) = 0.00;
	kernel.at<float>(0, 1) = 0.00; kernel.at<float>(1, 1) = 0.05; kernel.at<float>(2, 1) = 0.10; kernel.at<float>(3, 1) = 0.40; kernel.at<float>(4, 1) = 0.10; kernel.at<float>(5, 1) = 0.05; kernel.at<float>(6, 1) = 0.00;
	kernel.at<float>(0, 2) = 0.00; kernel.at<float>(1, 2) = 0.10; kernel.at<float>(2, 2) = 0.30; kernel.at<float>(3, 2) = 0.60; kernel.at<float>(4, 2) = 0.30; kernel.at<float>(5, 2) = 0.10; kernel.at<float>(6, 2) = 0.00;
	kernel.at<float>(0, 3) = 0.20; kernel.at<float>(1, 3) = 0.40; kernel.at<float>(2, 3) = 0.60; kernel.at<float>(3, 3) = 1.00; kernel.at<float>(4, 3) = 0.60; kernel.at<float>(5, 3) = 0.40; kernel.at<float>(6, 3) = 0.20;
	kernel.at<float>(0, 4) = 0.00; kernel.at<float>(1, 4) = 0.10; kernel.at<float>(2, 4) = 0.30; kernel.at<float>(3, 4) = 0.60; kernel.at<float>(4, 4) = 0.30; kernel.at<float>(5, 4) = 0.10; kernel.at<float>(6, 4) = 0.00;
	kernel.at<float>(0, 5) = 0.00; kernel.at<float>(1, 5) = 0.05; kernel.at<float>(2, 5) = 0.10; kernel.at<float>(3, 5) = 0.40; kernel.at<float>(4, 5) = 0.10; kernel.at<float>(5, 5) = 0.05; kernel.at<float>(6, 5) = 0.00;
	kernel.at<float>(0, 6) = 0.00; kernel.at<float>(1, 6) = 0.00; kernel.at<float>(2, 6) = 0.00; kernel.at<float>(3, 6) = 0.20; kernel.at<float>(4, 6) = 0.00; kernel.at<float>(5, 6) = 0.00; kernel.at<float>(6, 6) = 0.00;
	cv::filter2D(val, blurred, -1, kernel);

	if (!cv::imwrite((m_path / "image_").string() + std::to_string(i) + ".png", blurred))
	{
		std::cout << "Erreure dans la génération d'image" << std::endl;
	}
	(*m_cap) << blurred;
}