#pragma once

#include <vector>
#include <iostream>
#include <fstream>

#include "Simulator/PDTypeDef.hpp"
#include "Util/MathHelper.hpp"

// eigen vector accessor
#define block_vector(a) block<3, 1>(3 * (a), 0)

namespace PD {

typedef enum {
    MATERIAL_TYPE_COROT,
    MATERIAL_TYPE_StVK,
    MATERIAL_TYPE_NEOHOOKEAN_EXTEND_LOG,
    MATERIAL_TYPE_TOTAL_NUM
} MaterialType;

typedef enum {
    CONSTRAINT_TYPE_ATTACHMENT,
    CONSTRAINT_TYPE_SPRING,
    CONSTRAINT_TYPE_SPRING_BENDING,
    CONSTRAINT_TYPE_COLLISION,
    CONSTRAINT_TYPE_TET,
    CONSTRAINT_TYPE_NULL,
    CONSTRAINT_TYPE_TOTAL_NUM
} ConstraintType;

struct Constraint {
public:
    Constraint()
    {
        m_constraint_type = CONSTRAINT_TYPE_NULL;
    }
    Constraint(ConstraintType type)
        : m_constraint_type(type)
    {
    }
    Constraint(ConstraintType type, PDScalar stiffness)
        : m_constraint_type(type), m_stiffness(stiffness)
    {
    }
    Constraint(const Constraint& other)
        : m_constraint_type(other.m_constraint_type), m_stiffness(other.m_stiffness)
    {
    }
    ~Constraint() {}

    virtual void GetMaterialProperty(PDScalar& stiffness) { stiffness = m_stiffness; }
    virtual void GetMaterialProperty(MaterialType& type, PDScalar& mu, PDScalar& lambda, PDScalar& kappa) { std::cout << "Warning: reach <Constraint::GetMaterialProperty> base class virtual function." << std::endl; }
    virtual void SetMaterialProperty(PDScalar stiffness) { m_stiffness = stiffness; }
    virtual void SetMaterialProperty(MaterialType type, PDScalar mu, PDScalar lambda, PDScalar kappa, PDScalar laplacian_coeff) { std::cout << "Warning: reach <Constraint::SetMaterialProperty> base class virtual function." << std::endl; }
    virtual PDScalar ComputeLaplacianWeight()
    {
        std::cout << "Warning: reach <Constraint::ComputeLaplacianWeight> base class virtual function." << std::endl;
        return 0;
    }

    virtual bool VertexIncluded(unsigned int vi) { return false; }

    virtual PDScalar EvaluateEnergy(const PDVector& x)
    {
        std::cout << "Warning: reach <Constraint::EvaluatePotentialEnergy> base class virtual function." << std::endl;
        return 0;
    }
    virtual PDScalar GetEnergy()
    {
        std::cout << "Warning: reach <Constraint::GetPotentialEnergy> base class virtual function." << std::endl;
        return 0;
    }
    virtual void EvaluateGradient(const PDVector& x, PDVector& gradient) { std::cout << "Warning: reach <Constraint::EvaluateGradient> base class virtual function." << std::endl; }
    virtual void EvaluateGradient(const PDVector& x) { std::cout << "Warning: reach <Constraint::EvaluateGradient> base class virtual function." << std::endl; }
    virtual void GetGradient(PDVector& gradient) { std::cout << "Warning: reach <Constraint::GetGradient> base class virtual function." << std::endl; }
    virtual PDScalar EvaluateEnergyAndGradient(const PDVector& x, PDVector& gradient)
    {
        std::cout << "Warning: reach <Constraint::EvaluateEnergyAndGradient> base class virtual function." << std::endl;
        return 0;
    }
    virtual PDScalar EvaluateEnergyAndGradient(const PDVector& x)
    {
        std::cout << "Warning: reach <Constraint::EvaluateEnergyAndGradient> base class virtual function." << std::endl;
        return 0;
    }
    virtual PDScalar GetEnergyAndGradient(PDVector& gradient)
    {
        std::cout << "Warning: reach <Constraint::GetEnergyAndGradient> base class virtual function." << std::endl;
        return 0;
    }
    virtual void EvaluateHessian(const PDVector& x, std::vector<PDSparseMatrixTriplet>& hessian_triplets, bool definiteness_fix = false) { std::cout << "Warning: reach <Constraint::EvaluateHessian> base class virtual function." << std::endl; }
    virtual void EvaluateHessian(const PDVector& x, bool definiteness_fix = false) { std::cout << "Warning: reach <Constraint::EvaluateHessian> base class virtual function." << std::endl; }
    virtual void ApplyHessian(const PDVector& x, PDVector& b){}; // b = H*x, applying hessian for this block only;
    virtual void EvaluateFiniteDifferentHessian(const PDVector& x, std::vector<PDSparseMatrixTriplet>& hessian_triplets, int index = -1) { std::cout << "Warning: reach <Constraint::EvaluateFiniteDifferentHessian> base class virtual function." << std::endl; }
    virtual void EvaluateWeightedLaplacian(std::vector<PDSparseMatrixTriplet>& laplacian_triplets, int index = -1) { std::cout << "Warning: reach <Constraint::EvaluateWeightedLaplacian> base class virtual function." << std::endl; }
    virtual void EvaluateWeightedDiagonal(std::vector<PDSparseMatrixTriplet>& diagonal_triplets, int index = -1) { std::cout << "Warning: reach <Constraint::EvaluateWeightedDiagonal> base class virtual function." << std::endl; }
    virtual void EvaluateWeightedLaplacian1D(std::vector<PDSparseMatrixTriplet>& laplacian_1d_triplets, int index = -1) { std::cout << "Warning: reach <Constraint::EvaluateWeightedLaplacian1D> base class virtual function." << std::endl; }

    // accesser
    virtual PDScalar GetVolume(const PDVector& x) { return 0; }
    virtual void GetRotation(const PDVector& x, std::vector<EigenQuaternion>& rotations) {}

    // inline
    const ConstraintType& Type() { return m_constraint_type; }

public:
    ConstraintType m_constraint_type;
    PDScalar m_stiffness;

    // saved energy
    PDScalar m_energy;

    // for visualization and selection
public:
    virtual void WriteToFileOBJ(std::ofstream& outfile, int& existing_vertices)
    { /*do nothing*/
    }
    virtual void WriteToFileOBJHead(std::ofstream& outfile)
    { /*do nothing*/
    }
    virtual void WriteToFileOBJTet(std::ofstream& outfile)
    { /*do nothing*/
    }
    // virtual void DrawTet(const PDVector& x, const VBO& vbos)
    // { /*do nothing*/
    // }
    // virtual PDScalar RayConstraintIntersection() {return false;}
};

struct QNPDPenaltyConstraint : public Constraint {
public:
    QNPDPenaltyConstraint(int p0, const EigenVector3& fixedpoint, const EigenVector3& normal);
    QNPDPenaltyConstraint(PDScalar stiffness, int p0, const EigenVector3& fixedpoint, const EigenVector3& normal);
    QNPDPenaltyConstraint(const QNPDPenaltyConstraint& other);
    virtual ~QNPDPenaltyConstraint();

    bool IsActive(const PDVector& x);
    virtual PDScalar EvaluateEnergy(const PDVector& x);
    virtual PDScalar GetEnergy();
    virtual void EvaluateGradient(const PDVector& x, PDVector& gradient);
    virtual void EvaluateGradient(const PDVector& x);
    virtual void GetGradient(PDVector& gradient);
    virtual PDScalar EvaluateEnergyAndGradient(const PDVector& x, PDVector& gradient);
    virtual PDScalar EvaluateEnergyAndGradient(const PDVector& x);
    virtual PDScalar GetEnergyAndGradient(PDVector& gradient);
    virtual void EvaluateHessian(const PDVector& x, bool definiteness_fix = false);
    virtual void ApplyHessian(const PDVector& x, PDVector& b); // b = H*x, applying hessian for this block only;
    virtual void EvaluateHessian(const PDVector& x, std::vector<PDSparseMatrixTriplet>& hessian_triplets, bool definiteness_fix = false, int index = -1);
    virtual void EvaluateWeightedLaplacian(std::vector<PDSparseMatrixTriplet>& laplacian_triplets, int index = -1);
    virtual void EvaluateWeightedLaplacian1D(std::vector<PDSparseMatrixTriplet>& laplacian_1d_triplets, int index = -1);

public:
    int m_p0;
    EigenVector3 m_g;
    EigenVector3 m_fixed_point;
    EigenVector3 m_normal;
    EigenMatrix3 m_H;
};

struct QNPDTetConstraint : public Constraint {
    inline static int P_tet_ct = 0;

public:
    QNPDTetConstraint(unsigned int p1, unsigned int p2, unsigned int p3, unsigned int p4, PDVector& x);
    QNPDTetConstraint(const QNPDTetConstraint& other);
    virtual ~QNPDTetConstraint();

    virtual bool VertexIncluded(unsigned int vi)
    {
        for (unsigned int i = 0; i != 4; i++) { return vi == m_p[i]; }
        return false;
    }

    virtual void GetMaterialProperty(MaterialType& type, PDScalar& mu, PDScalar& lambda, PDScalar& kappa);
    virtual void SetMaterialProperty(MaterialType type, PDScalar mu, PDScalar lambda, PDScalar kappa, PDScalar laplacian_coeff);
    virtual PDScalar ComputeLaplacianWeight();

    virtual PDScalar EvaluateEnergy(const PDVector& x);
    virtual PDScalar GetEnergy();
    virtual void EvaluateGradient(const PDVector& x, PDVector& gradient);
    virtual void EvaluateGradient(const PDVector& x);
    virtual void GetGradient(PDVector& gradient);
    virtual PDScalar EvaluateEnergyAndGradient(const PDVector& x, PDVector& gradient);
    virtual PDScalar EvaluateEnergyAndGradient(const PDVector& x);
    virtual PDScalar GetEnergyAndGradient(PDVector& gradient);
    virtual void EvaluateHessian(const PDVector& x, bool definiteness_fix = false);
    virtual void EvaluateHessian(const PDVector& x, std::vector<PDSparseMatrixTriplet>& hessian_triplets, bool definiteness_fix = false);
    virtual void ApplyHessian(const PDVector& x, PDVector& b); // b = H*x, applying hessian for this block only;
    virtual void EvaluateFiniteDifferentHessian(const PDVector& x, std::vector<PDSparseMatrixTriplet>& hessian_triplets, int index = -1);
    virtual void EvaluateWeightedLaplacian(std::vector<PDSparseMatrixTriplet>& laplacian_triplets, int index = -1);
    virtual void EvaluateWeightedLaplacian1D(std::vector<PDSparseMatrixTriplet>& laplacian_1d_triplets, int index = -1);

    // accesser
    virtual PDScalar GetVolume(const PDVector& x);
    virtual void GetRotation(const PDVector& x, std::vector<EigenQuaternion>& rotations);

    // set mass matrix
    PDScalar SetMassMatrix(std::vector<PDSparseMatrixTriplet>& m, std::vector<PDSparseMatrixTriplet>& m_1d);

    // accesser
    inline const PDScalar& Volume() { return m_W; }

public:
    // for ghost force detection and visualization
    void KeyHessianAndRotation(const PDVector& x, bool definiteness_fix = false);
    void KeyHessianAndRotation(const PDVector& x, std::vector<PDSparseMatrixTriplet>& hessian_triplets, bool definiteness_fix = false);
    void KeyRotation(const PDVector& x);
    void GetRotationChange(const PDVector& x, std::vector<EigenQuaternion>& rotations);
    void ComputeGhostForce(const PDVector& x, const std::vector<EigenQuaternion>& rotations, bool stiffness_warped_matrix = true, bool outdated = true);
    virtual void WriteToFileOBJTet(std::ofstream& outfile, const PDVector& x);
    // virtual void DrawTet(const PDVector& x, const VBO& vbos);

private:
    void getMatrixDs(EigenMatrix3& Dd, const PDVector& x);
    void getDeformationGradient(EigenMatrix3& F, const PDVector& x);
    void getStressTensor(EigenMatrix3& P, const EigenMatrix3& F, EigenMatrix3& R);
    PDScalar getStressTensorAndEnergyDensity(EigenMatrix3& P, const EigenMatrix3& F, EigenMatrix3& R);
    void singularValueDecomp(EigenMatrix3& U, EigenVector3& SIGMA, EigenMatrix3& V, const EigenMatrix3& A, bool signed_svd = true);
    void extractRotation(EigenMatrix3& R, const EigenMatrix3& A, bool signed_svd = true);
    void evaluateFDGradient(const PDVector& x, PDVector& gradient);

    void calculateDPDF(Matrix3333& dPdF, const EigenMatrix3& F);

public:
    // material properties
    MaterialType m_material_type;
    PDScalar m_mu;
    PDScalar m_lambda;
    PDScalar m_kappa;
    PDScalar m_laplacian_coeff;

    unsigned int m_p[4]; // indices of four vertices
    EigenVector3 m_g[4];
    EigenMatrix12 m_H;
    // EigenMatrix3 m_H_blocks[4][4];
    EigenMatrix3 m_Dm; // [x1-x4|x2-x4|x3-x4]
    EigenMatrix3 m_Dm_inv; // inverse of m_Dm
    Eigen::Matrix<PDScalar, 3, 4> m_G; // G = m_Dr^(-T) * IND;
    PDScalar m_W; // 1/6 det(Dr);

    // for neohookean inverted part
    const PDScalar m_neohookean_clamp_value = 0.1;

    // for ghost force detection and visualization
    EigenMatrix12 m_previous_hessian;
    EigenMatrix3 m_previous_rotation;
    EigenVector3 m_ghost_force;
};

}