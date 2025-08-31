#pragma once
#include <cmath>
#include <cuda_runtime.h>

__device__ __host__
inline float ffmin(float a, float b) {
    return (a < b) ? a : b;
}

struct Vec3 {
    float x, y, z;

    __host__ __device__
        Vec3() : x(0.0f), y(0.0f), z(0.0f) {}

    __host__ __device__
        Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    // Addition
    __host__ __device__
        Vec3 operator+(const Vec3& other) const {
        return Vec3{ x + other.x, y + other.y, z + other.z };
    }

    __host__ __device__
        Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    // Subtraction
    __host__ __device__
        Vec3 operator-(const Vec3& other) const {
        return Vec3{ x - other.x, y - other.y, z - other.z };
    }

    __host__ __device__
        Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    // Scalar multiplication
    __host__ __device__
        Vec3 operator*(float scalar) const {
        return Vec3{ x * scalar, y * scalar, z * scalar };
    }

    __host__ __device__
        Vec3& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
    __host__ __device__
        bool operator==(const Vec3& v) const {
        return abs(x - v.x) < 1e-7f \
            && abs(y - v.x) < 1e-7f \
            && abs(z - v.x) < 1e-7f;
    }

    // Scalar division
    __host__ __device__
        Vec3 operator/(float scalar) const {
        return Vec3{ x / scalar, y / scalar, z / scalar };
    }

    __host__ __device__
        Vec3& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    // Norm (length)
    __host__ __device__
        float length() const {
        return std::sqrtf(x * x + y * y + z * z);
    }
    
    // Norm (length)
    __host__ __device__
        static float length(const Vec3& v) {
        return v.length();
    }

    // Normalize (returns a new vector)
    __host__ __device__
        Vec3 normalized() const {
        float len = length();
        if (len > 0.0) {
            return *this / len;
        }
        return Vec3{ 0.0, 0.0, 0.0 };
    }

    // Static dot product
    __host__ __device__
        static float dot(const Vec3& a, const Vec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    // Static normalize
    __host__ __device__
        static Vec3 normalize(const Vec3& v) {
        return v.normalized();
    }

    // Static distance
    __host__ __device__
        static float distance(const Vec3& a, const Vec3& b) {
        return (a - b).length();
    }
};

__host__ __device__
inline Vec3 operator*(float scalar, const Vec3& vec) {
    return Vec3{ vec.x * scalar, vec.y * scalar, vec.z * scalar };
}