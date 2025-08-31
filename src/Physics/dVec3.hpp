#pragma once
#include <cmath>
#include <cuda_runtime.h>

__device__ __host__
inline double dmin(double a, double b) {
    return (a < b) ? a : b;
}

struct dVec3 {
    double x, y, z;

    __host__ __device__
        dVec3() : x(0.0), y(0.0), z(0.0) {}

    __host__ __device__
        dVec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    // Addition
    __host__ __device__
        dVec3 operator+(const dVec3& other) const {
        return dVec3{ x + other.x, y + other.y, z + other.z };
    }

    __host__ __device__
        dVec3& operator+=(const dVec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    // Subtraction
    __host__ __device__
        dVec3 operator-(const dVec3& other) const {
        return dVec3{ x - other.x, y - other.y, z - other.z };
    }

    __host__ __device__
        dVec3& operator-=(const dVec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    // Scalar multiplication
    __host__ __device__
        dVec3 operator*(double scalar) const {
        return dVec3{ x * scalar, y * scalar, z * scalar };
    }

    __host__ __device__
        dVec3& operator*=(double scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    // Scalar division
    __host__ __device__
        dVec3 operator/(double scalar) const {
        return dVec3{ x / scalar, y / scalar, z / scalar };
    }

    __host__ __device__
        dVec3& operator/=(double scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    // Norm (length)
    __host__ __device__
        double length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    // Norm (length)
    __host__ __device__
        static double length(const dVec3& v) {
        return v.length();
    }

    // Normalize (returns a new vector)
    __host__ __device__
        dVec3 normalized() const {
        double len = length();
        if (len > 0.0) {
            return *this / len;
        }
        return dVec3{ 0.0, 0.0, 0.0 };
    }

    // Static dot product
    __host__ __device__
        static double dot(const dVec3& a, const dVec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    // Static normalize
    __host__ __device__
        static dVec3 normalize(const dVec3& v) {
        return v.normalized();
    }

    // Static distance
    __host__ __device__
        static double distance(const dVec3& a, const dVec3& b) {
        return (a - b).length();
    }
};

__host__ __device__
inline dVec3 operator*(double scalar, const dVec3& vec) {
    return dVec3{ vec.x * scalar, vec.y * scalar, vec.z * scalar };
}