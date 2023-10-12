// Copy from ipc

#include "MeshIO.hpp"

#include <ghc/fs_std.hpp> // filesystem
#include <spdlog/spdlog.h>
#include <mshio/mshio.h>

#include <boost/functional/hash.hpp>

class Triplet {
public:
    std::array<int, 3> key;

    Triplet(const int* p_key)
    {
        key[0] = p_key[0];
        key[1] = p_key[1];
        key[2] = p_key[2];
    }
    Triplet(int key0, int key1, int key2)
    {
        key[0] = key0;
        key[1] = key1;
        key[2] = key2;
    }

    int operator[](int i) const
    {
        assert(0 <= i && i <= 2);
        return key[i];
    }

    bool operator<(const Triplet& right) const
    {
        if (key[0] < right.key[0]) {
            return true;
        }
        else if (key[0] == right.key[0]) {
            if (key[1] < right.key[1]) {
                return true;
            }
            else if (key[1] == right.key[1]) {
                if (key[2] < right.key[2]) {
                    return true;
                }
            }
        }
        return false;
    }

    bool operator==(const Triplet& right) const
    {
        return key[0] == right[0] && key[1] == right[1] && key[2] == right[2];
    }
};

namespace std {
template <>
struct hash<Triplet> {
    size_t operator()(Triplet const& triplet) const noexcept
    {
        return boost::hash_range(triplet.key.begin(), triplet.key.end());
    }
};
}

inline std::ostream& operator<<(std::ostream& os, const Triplet& t)
{
    os << t.key[0] << ' ' << t.key[1] << ' ' << t.key[2];
    return os;
}

// ================================================================================================================

static void findSurfaceTris(const Eigen::MatrixXi& TT, Eigen::MatrixXi& F)
{
    // TODO: merge with below
    std::unordered_map<Triplet, int> tri2Tet(4 * TT.rows());
    for (int elemI = 0; elemI < TT.rows(); elemI++) {
        const Eigen::RowVector4i& elemVInd = TT.row(elemI);
        tri2Tet[Triplet(elemVInd[0], elemVInd[2], elemVInd[1])] = elemI;
        tri2Tet[Triplet(elemVInd[0], elemVInd[3], elemVInd[2])] = elemI;
        tri2Tet[Triplet(elemVInd[0], elemVInd[1], elemVInd[3])] = elemI;
        tri2Tet[Triplet(elemVInd[1], elemVInd[2], elemVInd[3])] = elemI;
    }

    // TODO: parallelize
    std::vector<Eigen::RowVector3i> tmpF;
    for (const auto& triI : tri2Tet) {
        const Triplet& triVInd = triI.first;
        // find dual triangle with reversed indices:
        bool isSurfaceTriangle = //
            tri2Tet.find(Triplet(triVInd[2], triVInd[1], triVInd[0])) == tri2Tet.end()
            && tri2Tet.find(Triplet(triVInd[1], triVInd[0], triVInd[2])) == tri2Tet.end()
            && tri2Tet.find(Triplet(triVInd[0], triVInd[2], triVInd[1])) == tri2Tet.end();
        if (isSurfaceTriangle) {
            tmpF.emplace_back(triVInd[0], triVInd[1], triVInd[2]);
        }
    }

    F.resize(tmpF.size(), 3);
    for (int i = 0; i < F.rows(); i++) {
        F.row(i) = tmpF[i];
    }
}

bool Util::readTetMesh(const std::string& filePath,
    Eigen::MatrixXd& TV, Eigen::MatrixXi& TT,
    Eigen::MatrixXi& F, bool findSurface)
{
    if (!fs::exists(filePath)) {
        return false;
    }

    mshio::MshSpec spec;
    try {
        spec = mshio::load_msh(filePath);
    }
    catch (...) {
        // MshIO only supports MSH 2.2 and 4.1 not 4.0
        return readTetMesh_msh4(filePath, TV, TT, F, findSurface);
    }

    const auto& nodes = spec.nodes;
    const auto& els = spec.elements;
    const int vAmt = nodes.num_nodes;
    int elemAmt = 0;
    for (const auto& e : els.entity_blocks) {
        assert(e.entity_dim == 3);
        assert(e.element_type == 4); // linear tet
        elemAmt += e.num_elements_in_block;
    }

    TV.resize(vAmt, 3);
    int index = 0;
    for (const auto& n : nodes.entity_blocks) {
        for (int i = 0; i < n.num_nodes_in_block * 3; i += 3) {
            TV.row(index) << n.data[i], n.data[i + 1], n.data[i + 2];
            ++index;
        }
    }

    TT.resize(elemAmt, 4);
    int elm_index = 0;
    for (const auto& e : els.entity_blocks) {
        for (int i = 0; i < e.data.size(); i += 5) {
            index = 0;
            for (int j = i + 1; j <= i + 4; ++j) {
                TT(elm_index, index++) = e.data[j] - 1;
            }
            ++elm_index;
        }
    }

    if (!findSurface) {
        spdlog::warn("readTetMesh is finding the surface because $Surface is not supported by MshIO");
    }
    spdlog::info("Finding the surface triangle mesh for {:s}", filePath);
    findSurfaceTris(TT, F);

    spdlog::info(
        "tet mesh loaded with {:d} nodes, {:d} tets, and {:d} surface triangles.",
        TV.rows(), TT.rows(), F.rows());

    return true;
}
bool Util::readTetMesh_msh4(const std::string& filePath,
    Eigen::MatrixXd& TV, Eigen::MatrixXi& TT,
    Eigen::MatrixXi& F, bool findSurface)
{
    FILE* in = fopen(filePath.c_str(), "r");
    if (!in) {
        return false;
    }

    TV.resize(0, 3);
    TT.resize(0, 4);
    F.resize(0, 3);

    char buf[BUFSIZ];
    while ((!feof(in)) && fgets(buf, BUFSIZ, in)) {
        if (strncmp("$Nodes", buf, 6) == 0) {
            fgets(buf, BUFSIZ, in);
            int vAmt;
            sscanf(buf, "1 %d", &vAmt);
            TV.resize(vAmt, 3);
            fgets(buf, BUFSIZ, in);
            break;
        }
    }
    assert(TV.rows() > 0);
    int bypass;
    for (int vI = 0; vI < TV.rows(); vI++) {
        fscanf(in, "%d %le %le %le\n", &bypass, &TV(vI, 0), &TV(vI, 1), &TV(vI, 2));
    }

    while ((!feof(in)) && fgets(buf, BUFSIZ, in)) {
        if (strncmp("$Elements", buf, 9) == 0) {
            fgets(buf, BUFSIZ, in);
            int elemAmt;
            sscanf(buf, "1 %d", &elemAmt);
            TT.resize(elemAmt, 4);
            fgets(buf, BUFSIZ, in);
            break;
        }
    }
    assert(TT.rows() > 0);
    for (int elemI = 0; elemI < TT.rows(); elemI++) {
        fscanf(in, "%d %d %d %d %d\n", &bypass,
            &TT(elemI, 0), &TT(elemI, 1), &TT(elemI, 2), &TT(elemI, 3));
    }
    TT.array() -= 1;

    while ((!feof(in)) && fgets(buf, BUFSIZ, in)) {
        if (strncmp("$Surface", buf, 7) == 0) {
            fgets(buf, BUFSIZ, in);
            int elemAmt;
            sscanf(buf, "%d", &elemAmt);
            F.resize(elemAmt, 3);
            break;
        }
    }
    for (int triI = 0; triI < F.rows(); triI++) {
        fscanf(in, "%d %d %d\n", &F(triI, 0), &F(triI, 1), &F(triI, 2));
    }
    if (F.rows() > 0) {
        F.array() -= 1;
    }
    else if (findSurface) {
        // if no surface triangles information provided, then find
        spdlog::info("Finding the surface triangle mesh for {:s}", filePath);
        findSurfaceTris(TT, F);
    }

    spdlog::info(
        "tet mesh loaded with {:d} nodes, {:d} tets, and {:d} surface triangles.",
        TV.rows(), TT.rows(), F.rows());

    fclose(in);

    return true;
}
void Util::readNodeEle(const std::string& filePath,
    Eigen::MatrixXd& TV, Eigen::MatrixXi& TT,
    Eigen::MatrixXi& F)
{
    FILE* in = fopen((filePath + ".node").c_str(), "r");
    if (!in) {
        spdlog::error("Unable to open node file: {:s}", filePath + ".node");
    }
    assert(in);

    int nN, nDim;
    fscanf(in, "%d %d 0 0", &nN, &nDim);
    spdlog::info("{:d} {:d}", nN, nDim);
    assert(nN >= 4);
    assert(nDim == 3);

    int bypass;
    TV.conservativeResize(nN, nDim);
    for (int vI = 0; vI < nN; ++vI) {
        fscanf(in, "%d %le %le %le", &bypass,
            &TV(vI, 0), &TV(vI, 1), &TV(vI, 2));
    }

    fclose(in);

    in = fopen((filePath + ".ele").c_str(), "r");
    assert(in);

    int nE, nDimp1;
    fscanf(in, "%d %d 0", &nE, &nDimp1);
    spdlog::info("{:d} {:d}", nE, nDimp1);
    assert(nE >= 0);
    assert(nDimp1 == 4);

    TT.conservativeResize(nE, nDimp1);
    for (int tI = 0; tI < nE; ++tI) {
        fscanf(in, "%d %d %d %d %d", &bypass,
            &TT(tI, 0), &TT(tI, 1), &TT(tI, 2), &TT(tI, 3));
    }

    fclose(in);

    findSurfaceTris(TT, F);

    std::cout << "tet mesh loaded with " << TV.rows() << " nodes, "
              << TT.rows() << " tets, and " << F.rows() << " surface tris." << std::endl;
}