#include "mbconfig.h"           // This goes first in every *.c,*.cc file

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>

#include "dataman.h"
#include "userelem.h"
#include "strnode.h" // StructNodeを使用するために必要
#include "drive.h"   // DriveOwnerを使用するために必要
#include "module-catenary_lm.h" // モジュール登録用のヘッダ

/*
 * =================================================================
 * === ModuleCatenaryLM クラス
 * =================================================================
*/
class ModuleCatenaryLM : virtual public Elem, public UserDefinedElem {
public:
    // コンストラクタとデストラクタ
    ModuleCatenaryLM(unsigned uLabel, const DofOwner *pDO, DataManager* pDM, MBDynParser& HP);
    virtual ~ModuleCatenaryLM(void);

    // MBDynインターフェース関数
    virtual void Output(OutputHandler& OH) const;
    virtual void WorkSpaceDim(integer* piNumRows, integer* piNumCols) const;

    SubVectorHandler&
    AssRes(SubVectorHandler& WorkVec,
        doublereal dCoef,
        const VectorHandler& XCurr, 
        const VectorHandler& XPrimeCurr);

    VariableSubMatrixHandler& 
    AssJac(VariableSubMatrixHandler& WorkMat,
        doublereal dCoef, 
        const VectorHandler& XCurr,
        const VectorHandler& XPrimeCurr);
    
    int iGetNumConnectedNodes(void) const;
    void GetConnectedNodes(std::vector<const Node *>& connectedNodes) const;

    // その他の必須仮想関数（今回は主に空実装）
    unsigned int iGetNumPrivData(void) const { return 0; };
    void SetValue(DataManager *pDM, VectorHandler& X, VectorHandler& XP, SimulationEntity::Hints *ph) { NO_OP; };
    std::ostream& Restart(std::ostream& out) const { return out << "# ModuleCatenaryLM: restart not implemented" << std::endl; };
    unsigned int iGetInitialNumDof(void) const { return 0; };
    void InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const {*piNumRows = 0; *piNumCols = 0;};
    VariableSubMatrixHandler& InitialAssJac(VariableSubMatrixHandler& WorkMat, const VectorHandler& XCurr) { WorkMat.SetNullMatrix(); return WorkMat; };
    SubVectorHandler& InitialAssRes(SubVectorHandler& WorkVec, const VectorHandler& XCurr) { WorkVec.ResizeReset(0); return WorkVec; };

private:
    // この要素が担当する、係留索を構成する全ノードへのポインタ
    std::vector<const StructNode*> m_nodes;

    // 物理パラメータ
    doublereal m_L_total;       // 係留索全長
    doublereal m_segment_length; // 1セグメントあたりの長さ
    doublereal m_EA;            // 軸剛性
    doublereal m_CA;            // 軸減衰
    doublereal m_rho_line;      // 水中線密度
    doublereal m_line_diameter; // 直径
    doublereal m_g_gravity;     // 重力加速度
    doublereal m_rho_water;     // 水の密度
    
    // 海底パラメータ
    doublereal m_seabed_z;      // 海底Z座標
    doublereal m_K_seabed;      // 海底剛性
    doublereal m_C_seabed;      // 海底減衰

    // 力のスケーリング用
    DriveOwner m_FSF;

    // 静的カテナリー適応
    bool bIsInitialStep;
    double m_catenary_w;
    double m_catenary_xacc;

    // --- 内部で使われる力計算関数 ---
    Vec3 ComputeAxialForce(const Vec3& x1, const Vec3& x2, const Vec3& v1, const Vec3& v2) const;
    Vec3 ComputeGravityBuoyancy(doublereal segment_mass) const;
    Vec3 ComputeSeabedForce(const Vec3& position, const Vec3& velocity) const;

    double myasinh(double X);
	double myacosh(double X);
	double myatanh(double X);

    void funcd(double x, double xacc, double &f, double &df, double d, double l, double &p0);
    double rtsafe(double x1, double x2, double xacc, double d, double l, double &p0);
};


// =================================================================
// === コンストラクタの実装
// =================================================================
ModuleCatenaryLM::ModuleCatenaryLM(
    unsigned uLabel,
    const DofOwner *pDO,
    DataManager* pDM,
    MBDynParser& HP
)
: Elem(uLabel, flag(0)),
  UserDefinedElem(uLabel, pDO),
  bIsInitialStep(true)
{

    if (HP.IsKeyWord("help")) {}

    m_L_total = HP.GetReal();
    m_rho_line = HP.GetReal();
    m_EA = HP.GetReal();
    m_CA = HP.GetReal();
    m_line_diameter = HP.GetReal();
    m_g_gravity = HP.GetReal();
    m_rho_water = HP.GetReal();

    if (HP.IsKeyWord("seabed")) {
        m_seabed_z = HP.GetReal();
        m_K_seabed = HP.GetReal();
        m_C_seabed = HP.GetReal();
    } else {
        m_seabed_z = -320;
        m_K_seabed = 0.0;
        m_C_seabed = 0.0;
    }

    int num_nodes = HP.GetInt();
    m_nodes.resize(num_nodes);

    for (int i = 0; i < num_nodes; ++i) {
        const Node* pNode = pDM->ReadNode(HP, Node::STRUCTURAL);
        
        if (!pNode) {
            silent_cerr("module-catenary_lm(" << GetLabel() << "): ERROR\n"
                << "    structural node expected but not found while reading node list.\n"
                << "    Check if all node labels in the .usr file are correctly defined in your .nod file.\n"
                << "    Error occurred reading " << i + 1 << "-th node in list at line: "
                << HP.GetLineData() << std::endl);
            throw ErrGeneric(MBDYN_EXCEPT_ARGS);
        }

        m_nodes[i] = dynamic_cast<const StructNode*>(pNode);

        if (!m_nodes[i]) {
            silent_cerr("module-catenary_lm(" << GetLabel() << "): ERROR\n"
                << "    Failed to cast node to StructNode.\n"
                << "    Node with label read from .usr file is not a structural node.\n"
                << "    Error occurred at line: " << HP.GetLineData() << std::endl);
            throw ErrGeneric(MBDYN_EXCEPT_ARGS);
        }
    }

    // 1セグメントあたりの長さを計算
    m_segment_length = m_L_total / (m_nodes.size() - 1);

    // Force Scale Factor
    if (HP.IsKeyWord("force" "scale" "factor")) {
        m_FSF.Set(HP.GetDriveCaller());
    } else {
        m_FSF.Set(new OneDriveCaller);
    }

    SetOutputFlag(pDM->fReadOutput(HP, Elem::LOADABLE));

    // ログファイルへの出力
    pDM->GetLogFile() << "catenary_lm: " << uLabel
        << " created, connected to " << m_nodes.size() << " nodes." << std::endl;
}

// デストラクタ
ModuleCatenaryLM::~ModuleCatenaryLM(void) {}


// =================================================================
// === MBDynインターフェースの実装
// =================================================================

// 接続ノード数の報告
int ModuleCatenaryLM::iGetNumConnectedNodes(void) const {
    return m_nodes.size();
}

// 接続ノードのポインタリストをMBDynに渡す
void ModuleCatenaryLM::GetConnectedNodes(std::vector<const Node *>& connectedNodes) const {
    connectedNodes.resize(m_nodes.size());
    for(size_t i = 0; i < m_nodes.size(); ++i) {
        connectedNodes[i] = m_nodes[i];
    }
}

// 作業領域のサイズをMBDynに伝える
void ModuleCatenaryLM::WorkSpaceDim(integer* piNumRows, integer* piNumCols) const {
    *piNumRows = m_nodes.size() * 6;
    *piNumCols = m_nodes.size() * 6;
}

SubVectorHandler&
ModuleCatenaryLM::AssRes(
    SubVectorHandler& WorkVec, 
    doublereal dCoef,
    const VectorHandler& XCurr, 
    const VectorHandler& XPrimeCurr)
{
    if (bIsInitialStep) {
        const StructNode* fairlead_node = m_nodes.front();
        const StructNode* anchor_node   = m_nodes.back();

        if (!fairlead_node || !anchor_node) {
            silent_cerr("ModuleCatenaryLM(" << GetLabel() << "): Fairlead or Anchor node is null in AssRes." << std::endl);
            throw ErrGeneric(MBDYN_EXCEPT_ARGS);
        }

        const Vec3& FP = fairlead_node->GetXCurr();
        const Vec3& AP = anchor_node->GetXCurr(); 

        Vec3 FP_AP = FP - AP;
        double h = std::fabs(FP_AP.dGet(3));
        double L_APFP = std::sqrt(std::pow(FP_AP.dGet(1), 2) + std::pow(FP_AP.dGet(2), 2));
        double H = 0.0; // 水平張力
        double V = 0.0; // 垂直張力

        if (h > 1e-9 && m_L_total > h && m_L_total > L_APFP) {
            double l = m_L_total / h;
            double d = (L_APFP - (m_L_total - h)) / h;

            if (d <= 0) {
                H = 0.0;
                V = m_catenary_w * h;
            } else if (d >= (std::sqrt(l*l - 1.0) - (l - 1.0))) {
                silent_cerr("ModuleCatenaryLM(" << GetLabel() << "): WARNING - Line length might be too short for catenary calculation, resulting in straight line tension." << std::endl);
                H = 0.0;
                V = 0.0;
            } else {
                try {
                    double p0 = 0.0;
                    double x1 = m_catenary_xacc;
                    double x2 = 1.0e+6;
                    double Ans_x = rtsafe(x1, x2, m_catenary_xacc, d, l, p0);

                    H = Ans_x * m_catenary_w * h;
                    V = m_catenary_w * h * std::sqrt(1.0 + 2.0 * Ans_x);
                } catch(const std::exception& e) {
                    silent_cerr("ModuleCatenaryLM(" << GetLabel() << "): rtsafe failed - " << e.what() << std::endl);
                    H = 0.0;
                    V = 0.0;
                }
            }
        }

        // フェアリーダーに作用する張力ベクトルを計算
        Vec3 F_tension(0.0, 0.0, 0.0);
        if (L_APFP > 1e-9) {
            double H_x = H * (FP_AP.dGet(1) / L_APFP);
            double H_y = H * (FP_AP.dGet(2) / L_APFP);
            // MBDynは力ベクトルを反作用として渡すため、フェアリーダーを「引く」方向、つまりアンカー側に向かうベクトルにする
            F_tension = Vec3(-H_x, -H_y, -V);
        } else {
            // 垂直に吊り下がっている場合
            F_tension = Vec3(0.0, 0.0, -V);
        }
        
        WorkVec.ResizeReset(6); // フェアリーダーノードの6自由度分だけ確保

        if (dynamic_cast<const DynamicStructNode*>(fairlead_node)) {
            integer iFirstIndex = fairlead_node->iGetFirstMomentumIndex();
            for (int j = 1; j <= 6; j++) { WorkVec.PutRowIndex(j, iFirstIndex + j); }
            WorkVec.Add(1, F_tension * m_FSF.dGet());
            WorkVec.Add(4, Vec3(0., 0., 0.));
        }
        
        bIsInitialStep = false;
    
    } else {

        int dynamic_dof_count = 0;
        for (size_t i = 0; i < m_nodes.size(); ++i) {
            if (dynamic_cast<const DynamicStructNode*>(m_nodes[i])) {
                dynamic_dof_count++;
            }
        }
        WorkVec.ResizeReset(dynamic_dof_count*6);

        int current_dof = 1;
        std::vector<int> node_to_dof_map(m_nodes.size(), -1); // マップを初期化
        for (size_t i = 0; i < m_nodes.size(); ++i) {
            if (dynamic_cast<const DynamicStructNode*>(m_nodes[i])) {
                node_to_dof_map[i] = current_dof; // ノードiがWorkVecの何番目から始まるか記録
                integer iFirstIndex = m_nodes[i]->iGetFirstMomentumIndex();
                for (int j = 1; j <= 6; j++) {
                    WorkVec.PutRowIndex(current_dof++, iFirstIndex + j);
                }
            }
        }

        doublereal dFSF = m_FSF.dGet();
        doublereal segment_mass = m_rho_line * m_segment_length;

        for (size_t i = 0; i < m_nodes.size() - 1; ++i) {
            const StructNode* node1 = m_nodes[i];
            const StructNode* node2 = m_nodes[i+1];

            const Vec3& x1 = node1->GetXCurr();
            const Vec3& x2 = node2->GetXCurr();
            const Vec3& v1 = node1->GetVCurr();
            const Vec3& v2 = node2->GetVCurr();

            Vec3 F_axial = ComputeAxialForce(x1, x2, v1, v2);
            Vec3 F_gravity_buoyancy = ComputeGravityBuoyancy(segment_mass);
            Vec3 F_seabed1 = ComputeSeabedForce(x1, v1);
            Vec3 F_seabed2 = ComputeSeabedForce(x2, v2);

            if (dynamic_cast<const DynamicStructNode*>(node1)) {
                Vec3 F1 = (-F_axial + F_gravity_buoyancy * 0.5 + F_seabed1) * dFSF;
                WorkVec.Add(i * 6 + 1, F1);
            }
            if (dynamic_cast<const DynamicStructNode*>(node2)) {
                Vec3 F2 = (F_axial + F_gravity_buoyancy * 0.5 + F_seabed2) * dFSF;
                WorkVec.Add((i + 1) * 6 + 1, F2);
            }
        }
    }

    return WorkVec;
}

VariableSubMatrixHandler& 
ModuleCatenaryLM::AssJac(
    VariableSubMatrixHandler& WorkMat, 
    doublereal dCoef, 
    const VectorHandler& XCurr, 
    const VectorHandler& XPrimeCurr)
{
    WorkMat.SetNullMatrix();
    return WorkMat;
}

void ModuleCatenaryLM::Output(OutputHandler& OH) const
{
    if (bToBeOutput()) {
        if (OH.UseText(OutputHandler::LOADABLE)) {
            // 例として、フェアリーダーとアンカーの張力を計算して出力
            const Vec3& fairlead_pos = m_nodes.front()->GetXCurr();
            const Vec3& node1_pos = m_nodes[1]->GetXCurr();
            const Vec3& fairlead_vel = m_nodes.front()->GetVCurr();
            const Vec3& node1_vel = m_nodes[1]->GetVCurr();

            const Vec3& last_vnode_pos = m_nodes[m_nodes.size() - 2]->GetXCurr();
            const Vec3& anchor_pos = m_nodes.back()->GetXCurr();
            const Vec3& last_vnode_vel = m_nodes[m_nodes.size() - 2]->GetVCurr();
            const Vec3& anchor_vel = m_nodes.back()->GetVCurr();
            
            Vec3 F_tension_fairlead = ComputeAxialForce(fairlead_pos, node1_pos, fairlead_vel, node1_vel);
            Vec3 F_tension_anchor = ComputeAxialForce(last_vnode_pos, anchor_pos, last_vnode_vel, anchor_vel);

            OH.Loadable() << GetLabel()
                << " fairlead_tension " << F_tension_fairlead.Norm()
                << " anchor_tension " << F_tension_anchor.Norm()
                << std::endl;
        }
    }
}


// =================================================================
// === 内部の物理計算関数の実装
// =================================================================

Vec3 ModuleCatenaryLM::ComputeAxialForce(const Vec3& x1, const Vec3& x2, const Vec3& v1, const Vec3& v2) const {
    Vec3 dx = x2 - x1;
    doublereal l_current = dx.Norm();
    
    if (l_current < 1e-12) {
        return Vec3(0.0, 0.0, 0.0);
    }
    
    Vec3 t = dx / l_current;
    doublereal strain = (l_current - m_segment_length) / m_segment_length;
    
    // 弾性力（引張のみ）
    doublereal F_elastic = (strain > 0.0) ? m_EA * strain : 0.0;
    
    // 減衰力
    Vec3 dv = v2 - v1;
    doublereal v_axial = dv.Dot(t);
    doublereal F_damping = m_CA * v_axial;
    
    doublereal F_total = F_elastic + F_damping;
    
    return t * F_total;
}

Vec3 ModuleCatenaryLM::ComputeGravityBuoyancy(doublereal segment_mass) const {
    doublereal weight = segment_mass * m_g_gravity;
    doublereal volume = (M_PI / 4.0) * m_line_diameter * m_line_diameter * m_segment_length;
    doublereal buoyancy = m_rho_water * volume * m_g_gravity;
    
    return Vec3(0.0, 0.0, -(weight - buoyancy));
}

Vec3 ModuleCatenaryLM::ComputeSeabedForce(const Vec3& position, const Vec3& velocity) const {
    Vec3 F_seabed(0.0, 0.0, 0.0);
    
    doublereal z = position.dGet(3);
    doublereal penetration = m_seabed_z - z;
    
    if (penetration > 0.0) {
        doublereal Fz = m_K_seabed * penetration - m_C_seabed * velocity.dGet(3);
        F_seabed = Vec3(0.0, 0.0, std::max(0.0, Fz)); // 上向きのみ
    }
    
    return F_seabed;
}

double ModuleCatenaryLM::myasinh(double X) { return std::log(X + std::sqrt(X * X + 1)); }
double ModuleCatenaryLM::myacosh(double X) { return std::log(X + std::sqrt(X + 1) * std::sqrt(X - 1)); }
double ModuleCatenaryLM::myatanh(double X) { return 0.5 * std::log((1 + X) / (1 - X)); }

void ModuleCatenaryLM::funcd(double x, double xacc, double& f, double& df, double d, double l, double& p0)
{
    int max = 1000;
    double f1, df1;

    if(x == 0.0) {
        f = -d;
        df = 0.0;
        p0 = 0.0;
    }
    else if(x > 0.0) {
        if(l <= 0.0) {
            double X_1 = 1.0/x + 1.0;
            f = x*myacosh(X_1) - std::sqrt(1.0 + 2.0*x) + 1.0 - d;
            df = myacosh(X_1) - 1.0/std::sqrt(1.0 + 2.0*x) - 1.0/(x*std::sqrt(std::pow(X_1, 2.0) - 1.0));
            p0 = 0.0;
        } else {
            if(x > (l*l - 1.0)/2) {
                p0 = 0.0;
                for(int i = 1; i < max; i++) {
                    double func1 = 1.0/x + 1.0/cos(p0);
                    f1 = x*(std::sqrt(std::pow(func1, 2.0) - 1.0) - std::tan(p0)) - l;
                    df1 = x*(func1*std::tan(p0)*(1.0/cos(p0))/std::sqrt(std::pow(func1, 2.0) - 1.0) - std::pow(std::tan(p0), 2.0) - 1.0);
                    p0 = p0 - f1/df1;
                    func1 = 1.0/x + 1.0/cos(p0);
                    f1 = x*(std::sqrt(std::pow(func1, 2.0) - 1.0) - std::tan(p0)) - l;
                    if(fabs(f1) < xacc) { break; }
                }

                if(fabs(f1) > xacc) {
                    std::cout << "fabs(f1)>eps" << std::endl;
                    throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
                }

                double X_2 = l/x + std::tan(p0);
                double X_3 = std::tan(p0);
                f = x*(myasinh(X_2) - myasinh(X_3)) - l + 1.0 - d;
                df = myasinh(X_2) - myasinh(X_3) - l/(x*std::sqrt(std::pow(X_2, 2.0) + 1.0));
            } else {
                double X_5 = 1.0/x + 1.0;
                f = x*myacosh(X_5) - std::sqrt(1.0 + 2.0*x) + 1.0 - d;
                df = myacosh(X_5) - 1.0/std::sqrt(1.0 + 2.0*x) - 1.0/(x*std::sqrt(std::pow(X_5, 2.0) - 1.0));
                p0 = 0.0;
            }
        }
    } else {
        std::cout << "ERROR (x<0)" << std::endl;
        throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
    }
}

double ModuleCatenaryLM::rtsafe(double x1, double x2, double xacc, double d, double l, double &p0)
{
	const int MAXIT = 1000;
    int j;
    double fh, fl, xh, xl, df;
    double dx, dxold, f, temp, rts;
    double p1, p2;

    funcd(x1, xacc, fl, df, d, l, p1);
    funcd(x2, xacc, fh, df, d, l, p2);

    if((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0)) {
        std::cout << "ERROR (fl>0.0&&fh>0.0)||(fl<0.0&&fh<0.0)" << std::endl;
        throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
    }

    if(fl == 0.0) { p0 = p1; return x1; }
    if(fh == 0.0) { p0 = p2; return x2; }

    if(fl < 0.0) { xl = x1; xh = x2; }
    else { xh = x1; xl = x2; }

    rts = 0.5*(x1 + x2);
    dxold = std::fabs(x2 - x1);
    dx = dxold;
    funcd(rts, xacc, f, df, d, l, p0);

    for(j = 0; j < MAXIT; j++) {
        if((((rts - xh)*df - f)*((rts - xl)*df - f) > 0.0) || ((std::fabs(2.0*f)) > std::fabs(dxold*df))) {
            dxold = dx;
            dx = 0.5*(xh - xl);
            rts = xl + dx;
            if(xl == rts) { return rts; }
        } else {
            dxold = dx;
            dx = f/df;
            temp = rts;
            rts -= dx;
            if(temp == rts) { return rts; }
        }

        if(std::fabs(dx) < xacc) { return rts; }

        funcd(rts, xacc, f, df, d, l, p0);

        if(f < 0.0) { xl = rts; }
        else { xh = rts; }
    }

    std::cout << "ERROR (Bisection method)" << std::endl;
    throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
}


// =================================================================
// === モジュール登録のための定型句
// =================================================================

bool catenary_lm_set(void) {
    #ifdef DEBUG
        std::cerr << __FILE__ << ":" << __LINE__ << ":" << __PRETTY_FUNCTION__ << std::endl;
    #endif

    UserDefinedElemRead *rf = new UDERead<ModuleCatenaryLM>;

    if (!SetUDE("catenary_lm", rf)) {
        delete rf;
        return false;
    }

    return true;
}

#ifndef STATIC_MODULES
extern "C" {
    int module_init(const char *module_name, void *pdm, void *php) {
        if (!catenary_lm_set()) {
            silent_cerr("catenary_lm: "
                "module_init(" << module_name << ") "
                "failed" << std::endl);
            return -1;
        }
        return 0;
    }
}
#endif // ! STATIC_MODULES
