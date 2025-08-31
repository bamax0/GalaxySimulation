#pragma once
#include "TypeDef.hpp"

// Distance
static constexpr DistanceM LIGHT_YEAR = 9460730472580800; // m
static constexpr DistanceM ASTRONOMICAL_UNIT = 149597870700; // m
static constexpr DistanceM SUN_DIAMATER = 696340000*2; // m
static constexpr DistanceM SOLAR_SYSTEM_DIAMATER = 200000* ASTRONOMICAL_UNIT; // m
static constexpr double SUN_DIAMATER_LY = SUN_DIAMATER / LIGHT_YEAR; // LY
static constexpr DistanceM MILKYWAY_SIZE = 100000 * LIGHT_YEAR; // m

// Mass
static constexpr double SOLAR_MASS = 1.9884e30; // Kg 
static constexpr MassKg MILKYWAY_MASS = 1.15e12 * SOLAR_MASS; // Kg

// Constant

static constexpr double LIGTH_SPEED = 299792458.; // m/s
static constexpr double INVERT_LIGTH_SPEED_SQUARED = 1.0/(LIGTH_SPEED* LIGTH_SPEED); // m^-2/s^-2
static constexpr double G = 6.67430e-11; // m3 kg-1 s-2
// static constexpr unsigned long long STARS_MILKYWAY{ static_cast<unsigned long>(400e9) }; // 400 milliard d'etoile
static constexpr TimeS UNIVERS_AGE = 4.32372e17; // s
static constexpr TimeS YEAR = 60.*60.*24.*365.; // s

static constexpr double REDUCED_G = G / (LIGHT_YEAR*LIGHT_YEAR*LIGHT_YEAR) * SOLAR_MASS * YEAR*YEAR; //  Ly3 Sm-1 Y-2
static constexpr float REDUCED_Gf = static_cast<float>(REDUCED_G); //  Ly3 Sm-1 Y-2
static constexpr float INVERT_LIGTH_SPEED_SQUAREDf = static_cast<float>(INVERT_LIGTH_SPEED_SQUARED); //  Ly3 Sm-1 Y-2

static constexpr float softening = 5.; // static_cast<float>(ASTRONOMICAL_UNIT / LIGHT_YEAR * 10000);
static constexpr float softening2 = softening *softening;

static constexpr float fPI = 3.14159265358979323846f;