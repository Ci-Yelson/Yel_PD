

#pragma once

#include "Simulator/PDTypeDef.hpp"

// ---------- COPY FROM Tiantian Liu (ltt1598@gmail.com) ----------

struct Matrix3333 // 3x3 matrix: each element is a 3x3 matrix
{
public:
    Matrix3333();
    Matrix3333(const Matrix3333& other);
    ~Matrix3333() {}

    void SetZero(); // [0 0 0; 0 0 0; 0 0 0]; 0 = 3x3 zeros
    void SetIdentity(); //[I 0 0; 0 I 0; 0 0 I]; 0 = 3x3 zeros, I = 3x3 identity

    // operators
    PD::EigenMatrix3& operator()(int row, int col);
    Matrix3333 operator+(const Matrix3333& plus);
    Matrix3333 operator-(const Matrix3333& minus);
    Matrix3333 operator*(const PD::EigenMatrix3& multi);
    friend Matrix3333 operator*(const PD::EigenMatrix3& multi1, Matrix3333& multi2);
    Matrix3333 operator*(PD::PDScalar multi);
    friend Matrix3333 operator*(PD::PDScalar multi1, Matrix3333& multi2);
    Matrix3333 transpose();
    PD::EigenMatrix3 Contract(const PD::EigenMatrix3& multi); // this operator is commutative
    Matrix3333 Contract(Matrix3333& multi);

protected:
    PD::EigenMatrix3 mat[3][3];
};

// matrix3333
inline Matrix3333::Matrix3333()
{
}

inline Matrix3333::Matrix3333(const Matrix3333& other)
{
    for (unsigned int row = 0; row != 3; ++row) {
        for (unsigned int col = 0; col != 3; ++col) {
            mat[row][col] = other.mat[row][col];
        }
    }
}

inline void Matrix3333::SetZero()
{
    for (unsigned int row = 0; row != 3; ++row) {
        for (unsigned int col = 0; col != 3; ++col) {
            mat[row][col] = PD::EigenMatrix3::Zero();
        }
    }
}

inline void Matrix3333::SetIdentity()
{
    for (unsigned int row = 0; row != 3; ++row) {
        for (unsigned int col = 0; col != 3; ++col) {
            mat[row][col] = PD::EigenMatrix3::Zero();
            mat[row][col](row, col) = 1.0;
        }
    }
}

inline PD::EigenMatrix3& Matrix3333::operator()(int row, int col)
{
    assert(row >= 0 && row < 3 && col >= 0 && col < 3);
    return mat[row][col];
}

inline Matrix3333 Matrix3333::operator+(const Matrix3333& plus)
{
    Matrix3333 res;
    for (unsigned int row = 0; row != 3; ++row) {
        for (unsigned int col = 0; col != 3; ++col) {
            res.mat[row][col] = mat[row][col] + plus.mat[row][col];
        }
    }
    return res;
}

inline Matrix3333 Matrix3333::operator-(const Matrix3333& minus)
{
    Matrix3333 res;
    for (unsigned int row = 0; row != 3; ++row) {
        for (unsigned int col = 0; col != 3; ++col) {
            res.mat[row][col] = mat[row][col] - minus.mat[row][col];
        }
    }
    return res;
}

inline Matrix3333 Matrix3333::operator*(const PD::EigenMatrix3& multi)
{
    Matrix3333 res;
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            res.mat[i][j].setZero();
            for (unsigned int k = 0; k < 3; k++) {
                res.mat[i][j] += mat[i][k] * multi(k, j);
            }
        }
    }
    return res;
}

inline Matrix3333 operator*(const PD::EigenMatrix3& multi1, Matrix3333& multi2)
{
    Matrix3333 res;
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            res(i, j).setZero();
            for (unsigned int k = 0; k < 3; k++) {
                res(i, j) += multi1(i, k) * multi2(k, j);
            }
        }
    }
    return res;
}

inline Matrix3333 Matrix3333::operator*(PD::PDScalar multi)
{
    Matrix3333 res;
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            res.mat[i][j] = mat[i][j] * multi;
        }
    }
    return res;
}

inline Matrix3333 operator*(PD::PDScalar multi1, Matrix3333& multi2)
{
    Matrix3333 res;
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            res(i, j) = multi1 * multi2(i, j);
        }
    }
    return res;
}

inline Matrix3333 Matrix3333::transpose()
{
    Matrix3333 res;
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            res(i, j) = mat[j][i];
        }
    }
    return res;
}

inline PD::EigenMatrix3 Matrix3333::Contract(const PD::EigenMatrix3& multi)
{
    PD::EigenMatrix3 res;
    res.setZero();
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            res += mat[i][j] * multi(i, j);
        }
    }
    return res;
}

inline Matrix3333 Matrix3333::Contract(Matrix3333& multi)
{
    Matrix3333 res;
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            res(i, j) = this->Contract(multi(i, j));
        }
    }
    return res;
}

inline void directProduct(Matrix3333& dst, const PD::EigenMatrix3& src1, const PD::EigenMatrix3& src2)
{
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            dst(i, j) = src1(i, j) * src2;
        }
    }
}
// void directProduct(Matrix2222& dst, const EigenMatrix2& src1, const EigenMatrix2& src2)
// {
// 	for (unsigned int i = 0; i < 2; ++i)
// 	{
// 		for (unsigned int j = 0; j < 2; ++j)
// 		{
// 			dst(i, j) = src1(i, j) * src2;
// 		}
// 	}
// }

// ----------------------------------------------------------------

// -------------- COPY FROM Huamin Wang --------------

template <class TYPE>
void Get_Rotation(TYPE F[3][3], TYPE R[3][3])
{
    TYPE C[3][3];
    memset(&C[0][0], 0, sizeof(TYPE) * 9);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                C[i][j] += F[k][i] * F[k][j];

    TYPE C2[3][3];
    memset(&C2[0][0], 0, sizeof(TYPE) * 9);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                C2[i][j] += C[i][k] * C[j][k];

    TYPE det = F[0][0] * F[1][1] * F[2][2] + F[0][1] * F[1][2] * F[2][0] + F[1][0] * F[2][1] * F[0][2] - F[0][2] * F[1][1] * F[2][0] - F[0][1] * F[1][0] * F[2][2] - F[0][0] * F[1][2] * F[2][1];

    TYPE I_c = C[0][0] + C[1][1] + C[2][2];
    TYPE I_c2 = I_c * I_c;
    TYPE II_c = 0.5 * (I_c2 - C2[0][0] - C2[1][1] - C2[2][2]);
    TYPE III_c = det * det;
    TYPE k = I_c2 - 3 * II_c;

    TYPE inv_U[3][3];
    if (k < 1e-10f) {
        TYPE inv_lambda = 1 / sqrt(I_c / 3);
        memset(inv_U, 0, sizeof(TYPE) * 9);
        inv_U[0][0] = inv_lambda;
        inv_U[1][1] = inv_lambda;
        inv_U[2][2] = inv_lambda;
    }
    else {
        TYPE l = I_c * (I_c * I_c - 4.5 * II_c) + 13.5 * III_c;
        TYPE k_root = sqrt(k);
        TYPE value = l / (k * k_root);
        if (value < -1.0) value = -1.0;
        if (value > 1.0) value = 1.0;
        TYPE phi = acos(value);
        TYPE lambda2 = (I_c + 2 * k_root * cos(phi / 3)) / 3.0;
        TYPE lambda = sqrt(lambda2);

        TYPE III_u = sqrt(III_c);
        if (det < 0) III_u = -III_u;
        TYPE I_u = lambda + sqrt(-lambda2 + I_c + 2 * III_u / lambda);
        TYPE II_u = (I_u * I_u - I_c) * 0.5;

        TYPE U[3][3];
        TYPE inv_rate, factor;

        inv_rate = 1 / (I_u * II_u - III_u);
        factor = I_u * III_u * inv_rate;

        memset(U, 0, sizeof(TYPE) * 9);
        U[0][0] = factor;
        U[1][1] = factor;
        U[2][2] = factor;

        factor = (I_u * I_u - II_u) * inv_rate;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                U[i][j] += factor * C[i][j] - inv_rate * C2[i][j];

        inv_rate = 1 / III_u;
        factor = II_u * inv_rate;
        memset(inv_U, 0, sizeof(TYPE) * 9);
        inv_U[0][0] = factor;
        inv_U[1][1] = factor;
        inv_U[2][2] = factor;

        factor = -I_u * inv_rate;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                inv_U[i][j] += factor * U[i][j] + inv_rate * C[i][j];
    }

    memset(&R[0][0], 0, sizeof(TYPE) * 9);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                R[i][j] += F[i][k] * inv_U[k][j];
}