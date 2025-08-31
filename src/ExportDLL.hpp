#pragma once
#pragma warning(disable: 4251 4275)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)

#ifdef _WIN32
    #if GALAXY_SIM_COMPILING_DLL
        #define  GALAXY_SIM_DLL_EXPORT __declspec(dllexport)
    #else
        #define  GALAXY_SIM_DLL_EXPORT __declspec(dllimport)
    #endif
#else
    #define GALAXY_SIM_DLL_EXPORT
#endif