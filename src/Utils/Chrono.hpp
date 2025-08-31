#pragma once
#include <chrono>
#include <iostream>
#include "ExportDLL.hpp"

// TODO faire un utils pour le mettre
using hcc = std::chrono::high_resolution_clock;
class GALAXY_SIM_DLL_EXPORT TestChrono
{
public:
	static void start()
	{
		m_start = hcc::now();
	}
	static void end()
	{
		m_end = hcc::now();
	}

	static double getTime()
	{
		std::chrono::duration<double, std::milli> duration_ms = m_end - m_start;
		return duration_ms.count();
	}

	static void show()
	{
		std::cout << "Execution time: " << getTime() << " ms" << std::endl;
	}

	static hcc::time_point m_start;
	static hcc::time_point m_end;
};
hcc::time_point TestChrono::m_start{};
hcc::time_point TestChrono::m_end{};