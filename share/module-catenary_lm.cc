#include "mbconfig.h"           // This goes first in every *.c,*.cc file

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>


#include "strnode.h" // 赤線：エディタの設定問題なので無視して OK
#include "dofown.h"
#include "dataman.h" // 赤線：Windows 環境で Unix 系ヘッダを参照しようとしているが，実際は MBDyn 側で条件分岐するので無視で OK
#include "userelem.h"
#include "module-catenary_lm.h" // 自前のヘッダ


// ========================================
// =========== セグメント要素クラス =========
// ========================================

class CatenarySegmentElement : virtual public Elem, public UserDefinedElem {
    private:
        const StructNode* node1; // セグメント始点ノード
        const StructNode* node2; // セグメント終点ノード

        // セグメント計算に必要なパラメータ
        doublereal EA;
        doublereal CA;
        doublereal L0; // セグメントの乾燥時自然長
        doublereal segment_mass; // セグメントの乾燥時質量
        doublereal line_diameter;
        doublereal rho_line;
        doublereal rho_water;
        doublereal g_gravity;

        // 海底パラメータ
        doublereal seabed_z;
        doublereal K_seabed;
        doublereal C_seabed;

        DriveOwner FSF;

    public:
        CatenarySegmentElement(
            unsigned uLabel, const DofOwner* pDO,
            const StructNode* n1, const StructNode* n2,
            doublereal ea, doublereal ca, doublereal l0,
            doublereal mass, doublereal diameter,
            doublereal rho_l, doublereal rho_w, doublereal gravity,
            doublereal sb_z, doublereal k_sb, doublereal c_sb,
            DriveCaller* pFSF
        );

        virtual ~CatenarySegmentElement(void);

        // MBDyn インターフェース
        virtual void Output(OutputHandler& OH) const;
        virtual void WorkSpaceDim(integer* piNumRows, integer* piNumCols) const;

        SubVectorHandler& AssRes(
            SubVectorHandler& WorkVec,
            doublereal dCoef,
            const VectorHandler& XCurr,
            const VectorHandler& XPrimeCurr
        );

        VariableSubMatrixHandler& AssJac(
            VariableSubMatrixHandler& WorkMat,
            doublereal dCoef,
            const VectorHandler& XCurr,
            const VectorHandler& XPrimeCurr
        );

        unsigned int iGetNumPrivData(void) const;
        int iGetNumConnectedNodes(void) const;
        void GetConnectedNodes(std::vector<const Node*>& connectedNodes) const;
        void SetValue(
            DataManager *pDM, 
            VectorHandler& X, 
            VectorHandler& XP,
            SimulationEntity::Hints *ph
        );
        std::ostream& Restart(std::ostream& out) const;

        // 初期化関連
        virtual unsigned int iGetInitialNumDof(void) const;
        virtual void InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const;
        VariableSubMatrixHandler& InitialAssJac(VariableSubMatrixHandler& WorkMat, const VectorHandler& XCurr);
        SubVectorHandler& InitialAssRes(SubVectorHandler& WorkVec, const VectorHandler& XCurr);

        Vec3 GetCurrentAxialForce() const {
            const Vec3& x1 = node1->GetXCurr();
            const Vec3& x2 = node2->GetXCurr();
            const Vec3& v1 = node1->GetVCurr();
            const Vec3& v2 = node2->GetVCurr();
            return ComputeAxialForce(x1, x2, v1, v2);
        }
    
        Vec3 GetCurrentGravityBuoyancy() const {
            return ComputeGravityBuoyancy();
        }
    
        Vec3 GetCurrentSeabedForce(bool for_node1 = true) const {
            const Vec3& pos = for_node1 ? node1->GetXCurr() : node2->GetXCurr();
            const Vec3& vel = for_node1 ? node1->GetVCurr() : node2->GetVCurr();
            return ComputeSeabedForce(pos, vel);
        }
    
        Vec3 GetCurrentFluidForce(bool for_node1 = true) const {
            const Vec3& pos = for_node1 ? node1->GetXCurr() : node2->GetXCurr();
            const Vec3& vel = for_node1 ? node1->GetVCurr() : node2->GetVCurr();
            return ComputeFluidForce(pos, vel);
        }
    
        doublereal GetCurrentLength() const {
            return (node2->GetXCurr() - node1->GetXCurr()).Norm();
        }
    
        doublereal GetCurrentStrain() const {
            doublereal current_length = GetCurrentLength();
            return (current_length - L0) / L0;
        }
    
        // パラメータアクセス用
        doublereal GetL0() const { return L0; }
        doublereal GetEA() const { return EA; }
        doublereal GetCA() const { return CA; }
        doublereal GetSegmentMass() const { return segment_mass; }
        doublereal GetLineDiameter() const { return line_diameter; }
        doublereal GetRhoLine() const { return rho_line; }
        doublereal GetRhoWater() const { return rho_water; }
        doublereal GetGravity() const { return g_gravity; }
        doublereal GetSeabedZ() const { return seabed_z; }
        doublereal GetKSeabed() const { return K_seabed; }
        doublereal GetCSeabed() const { return C_seabed; }
    
        const StructNode* GetNode1() const { return node1; }
        const StructNode* GetNode2() const { return node2; }

    private:
        // 力計算メソッド
        Vec3 ComputeAxialForce(const Vec3& x1, const Vec3& x2, const Vec3& v1, const Vec3& v2) const;
        Vec3 ComputeGravityBuoyancy() const;
        Vec3 ComputeSeabedForce(const Vec3& position, const Vec3& velocity) const;
        Vec3 ComputeFluidForce(const Vec3& position, const Vec3& velocity) const;
    
        // 剛性・減衰行列計算
        void ComputeStiffnessMatrix(const Vec3& x1, const Vec3& x2, Mat3x3& K11, Mat3x3& K12, Mat3x3& K21, Mat3x3& K22);
        void ComputeDampingMatrix(const Vec3& x1, const Vec3& x2, Mat3x3& C11, Mat3x3& C12, Mat3x3& C21, Mat3x3& C22);
        Mat3x3 ComputeSeabedStiffness(const Vec3& position);
        Mat3x3 ComputeSeabedDamping(const Vec3& position);
};

// ===============================
// ======== メインクラス ==========
//================================

class ModuleCatenaryLM : virtual public Elem, public UserDefinedElem {
    public:
        // コンストラクタとデストラクタ
        ModuleCatenaryLM(unsigned uLabel, const DofOwner *pDO, DataManager* pDM, MBDynParser& HP);
        virtual ~ModuleCatenaryLM(void);

        // MBDyn インターフェース
        virtual void Output(OutputHandler& OH) const;
        virtual void WorkSpaceDim(integer* piNumRows, integer* piNumCols) const;

        VariableSubMatrixHandler& AssJac(
            VariableSubMatrixHandler& WorkMat,
            doublereal dCoef,
            const VectorHandler& XCurr,
            const VectorHandler& XPrimeCurr
        );

        SubVectorHandler& AssRes(
            SubVectorHandler& WorkVec,
            doublereal dCoef,
            const VectorHandler& XCurr,
            const VectorHandler& XPrimeCurr
        );

        // データアクセスメソッド
        unsigned int iGetNumPrivData(void) const;
        int iGetNumConnectedNodes(void) const;
        void GetConnectedNodes(std::vector<const Node *>& connectedNodes) const;
        void SetValue(
            DataManager *pDM, 
            VectorHandler& X, 
            VectorHandler& XP,
            SimulationEntity::Hints *ph
        );
        std::ostream& Restart(std::ostream& out) const;

        // 初期化関連メソッド
        virtual unsigned int iGetInitialNumDof(void) const;
        virtual void InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const;
        VariableSubMatrixHandler& InitialAssJac(
            VariableSubMatrixHandler& WorkMat,
            const VectorHandler& XCurr
        );
        SubVectorHandler& InitialAssRes(
            SubVectorHandler& WorkVec, 
            const VectorHandler& XCurr
        );

    private:
        const StructNode* g_pNode;          // フェアリーダーを表す MBDyn の構造ノード
        
        // 基本パラメータ
        double APx, APy, APz;               // アンカーのグローバル x,y,z 座標
        double L;                           // ライン全長
        double w;                           // チェーンの単重
        double xacc;                        // rtsafe精度

        // ランプドマス法パラメータ
        doublereal EA;                      // 軸剛性
        doublereal CA;                      // 軸減衰
        doublereal rho_line;                // 線密度
        doublereal line_diameter;           // 線径
        doublereal g_gravity;               // 重力加速度
        doublereal rho_water;               // 水の密度

        // 海底パラメータ
        doublereal seabed_z;                // 海底z座標
        doublereal K_seabed;                // 海底剛性
        doublereal C_seabed;                // 海底減衰

        DriveOwner FSF;                     // Force Scale Factor

        // 仮想ノード　要素管理
        std::vector<DynamicStructNode*> virtual_nodes; // MBDyn 動的ノード
        std::vector<CatenarySegmentElement*> segment_elements; // セグメント要素
        static const unsigned int NUM_SEGMENTS = 20; // セグメント数

        // カテナリー理論関数
        static double myasinh(double X);
        static double myacosh(double X);
        static void funcd(double x, double xacc, double &f, double &df, double d, double l, double &p0);
        static double rtsafe(double x1, double x2, double xacc, double d, double l, double &p0);

        // 初期化メソッド
        void CreateVirtualNodes(DataManager* pDM);
        void CreateSegmentElements(DataManager* pDM);
        void InitializeVirtualNodesFromCatenary();
        const StructNode* CreateAnchorNode(DataManager* pDM);

        std::vector<Vec3> initial_positions;  // 初期位置保存用
};

// =============================
// ========= コンストラ =========
// =============================

ModuleCatenaryLM::ModuleCatenaryLM(
    unsigned uLabel,
    const DofOwner *pDO,
    DataManager* pDM,
    MBDynParser& HP
)
    : Elem(uLabel, flag(0)),
      UserDefinedElem(uLabel, pDO),
      g_pNode(nullptr),
      APx(0), APy(0), APz(0),
      L(0), w(0), xacc(1e-6),
      EA(3.842e8), CA(0.0),
      rho_line(77.71), line_diameter(0.09017),
      g_gravity(9.80665), rho_water(1025.0),
      seabed_z(-320.0), K_seabed(1.0e5), C_seabed(1.0e3)
{
    // help
    if (HP.IsKeyWord("help")) {
        silent_cout(
            "\n"
            "Module: 	ModuleCatenaryLM (Lumped Mass Method)\n"
            "Usage:      catenary_lm, fairlead_node_label, \n"
            "                  LineLength, total_length,\n"
            "                  LineWeight, unit_weight,\n"
            "                  Xacc, rtsafe_accuracy,\n"
            "                  APx, APy, APz,\n"
            "                [ EA, axial_stiffness, ]\n"
            "                [ CA, axial_damping, ]\n"
            "                [ rho_line, line_density, ]\n"
            "                [ line_diameter, diameter, ]\n"
            "                [ gravity, g_acceleration, ]\n"
            "                [ water_density, rho_w, ]\n" // オプション追加
            "                [ seabed, z_coordinate, k_stiffness, c_damping, ]\n"
            "                [ ramp_time, ramp_duration, ]\n"
            "                [ force scale factor, (DriveCaller), ]\n"
            "                [ output, (FLAG) ] ;\n"
            "\n"
            << std::endl
        );

        if (!HP.IsArg()) {
            throw NoErr(MBDYN_EXCEPT_ARGS);
        }
    }

    // MBDyn のパーサー HP を介して各パラメータをメンバ変数に格納する
    g_pNode = dynamic_cast<const StructNode *>(pDM->ReadNode(HP, Node::STRUCTURAL));
    L = HP.GetReal();
    w = HP.GetReal();
    xacc = HP.GetReal();
    APx = HP.GetReal();
    APy = HP.GetReal();
    APz = HP.GetReal();

    // オプショナルパラメータ
    if (HP.IsKeyWord("EA")) { EA = HP.GetReal(); }
    if (HP.IsKeyWord("CA")) { CA = HP.GetReal(); }
    if (HP.IsKeyWord("rho_line")) { rho_line = HP.GetReal(); }
    if (HP.IsKeyWord("line_diameter")) { line_diameter = HP.GetReal(); }
    if (HP.IsKeyWord("gravity")) { g_gravity = HP.GetReal(); }
    if (HP.IsKeyWord("water_density")) { rho_water = HP.GetReal(); } // オプション追加
    if (HP.IsKeyWord("seabed")) {
        seabed_z = HP.GetReal();
        K_seabed = HP.GetReal();
        C_seabed = HP.GetReal();
    }

    // Force Scale Factor
    if (HP.IsKeyWord("Force" "scale" "factor")) {
        FSF.Set(HP.GetDriveCaller());
    } else {
        FSF.Set(new OneDriveCaller);
    }

    SetOutputFlag(pDM->fReadOutput(HP, Elem::LOADABLE));

    // 仮想ノード初期化
    CreateVirtualNodes(pDM);

    // カテナリー理論による初期位置設定
    InitializeVirtualNodesFromCatenary();

    // セグメント要素作成
    CreateSegmentElements(pDM);

    pDM->GetLogFile() << "catenary_lm: " << uLabel
        << " created" << virtual_nodes.size() << " virtual nodes and"
        << segment_elements.size() << " segment elements" << std::endl;
}

ModuleCatenaryLM::~ModuleCatenaryLM(void)
{
    // DataManager 自動的にノード・要素を削除するため明示的な削除は不要
}

// =================================
// ========= 仮想ノード作成 =========
// =================================

void ModuleCatenaryLM::CreateVirtualNodes(DataManager* pDM)
{
    virtual_nodes.resize(NUM_SEGMENTS - 1);

    for (unsigned int i = 0; i < NUM_SEGMENTS - 1; ++i) {
        // 仮想ノードのラベル作成
        unsigned int node_label = GetLabel()*1000 + i + 1;

        // 一時的な位置で作成し，後にカテナリー理論に基づき初期位置の更新
        Vec3 temp_pos(0.0, 0.0, 0.0);
        doublereal node_mass = (rho_line * L) / static_cast<doublereal>(NUM_SEGMENTS);

        virtual_nodes[i] = new DynamicStructNode(
            node_label, // ノードラベル
            pGetDofOwner(), // DofOwner
            temp_pos, // 初期位置：一時的に
            Mat3x3(Eye3), // 初期姿勢：単位行列
            Vec3(Zero3), // 初期速度：ゼロ
            Vec3(Zero3), // 初期角速度：ゼロ
            static_cast<const StructNode*>(nullptr),  // 参照ノード：なし
            static_cast<const RigidBodyKinematics*>(nullptr), // 姿勢記述
            node_mass,                     // ノード質量
            0.0,
            false,
            OrientationDescription::EULER_321,
            flag(0)
        );

        // DataManager に登録
        pDM->Add(virtual_nodes[i]);
    }
}

// ===========================
// === アンカーノード作成 ======
// ==========================

const StructNode* ModuleCatenaryLM::CreateAnchorNode(DataManager* pDM)
{
    unsigned int anchor_label = GetLabel() * 1000 + 999;
    Vec3 anchor_pos(APx, APy, APz);
    
    // 静的ノードとしてアンカーを作成
    StaticStructNode* anchor_node = new StaticStructNode(
        anchor_label,
        pGetDofOwner(),
        anchor_pos,
        Mat3x3(Eye3),
        Vec3(Zero3),
        Vec3(Zero3),
        static_cast<const StructNode*>(nullptr),
        static_cast<const RigidBodyKinematics*>(nullptr),
        0.0,
        0.0,
        false,
        OrientationDescription::EULER_321,
        flag(0)
    );
    
    pDM->Add(anchor_node);
    
    return anchor_node;
}

// ====================================
// ========= セグメント要素作成 =========
// ====================================

void ModuleCatenaryLM::CreateSegmentElements(DataManager* pDM)
{
    // アンカーノード作成
    const StructNode* anchor_node = CreateAnchorNode(pDM);
    segment_elements.resize(NUM_SEGMENTS);
    
    for (unsigned int i = 0; i < NUM_SEGMENTS; ++i) {
        const StructNode* n1 = (i == 0) ? g_pNode : virtual_nodes[i-1];
        const StructNode* n2 = (i == NUM_SEGMENTS-1) ? anchor_node : virtual_nodes[i];
        
        unsigned int elem_label = GetLabel() * 1000 + 100 + i;
        doublereal segment_length = L / static_cast<doublereal>(NUM_SEGMENTS);
        doublereal segment_mass = (rho_line * L) / static_cast<doublereal>(NUM_SEGMENTS);
        
        segment_elements[i] = new CatenarySegmentElement(
            elem_label, pGetDofOwner(), n1, n2,
            EA, CA, segment_length, segment_mass, line_diameter,
            rho_line, rho_water, g_gravity,
            seabed_z, K_seabed, C_seabed,
            FSF.pGetDriveCaller()
        );
        
        pDM->Add(segment_elements[i]);
    }
}

// ========= カテナリー理論関数 =========
double ModuleCatenaryLM::myasinh(double X)
{
    return std::log(X + std::sqrt(X * X + 1));
}

double ModuleCatenaryLM::myacosh(double X)
{
    return std::log(X + std::sqrt(X + 1) * std::sqrt(X - 1));
}

void ModuleCatenaryLM::funcd(double x, double xacc, double& f, double& df, double d, double l, double &p0)
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

// ==========================================================
// ========= カテナリー理論による仮想ノード初期位置設定 =========
// =========================================================

void ModuleCatenaryLM::InitializeVirtualNodesFromCatenary()
{
    if (virtual_nodes.empty()) {
        return;
    }

    const Vec3& FP = g_pNode->GetXCurr();
    Vec3 AP(APx, APy, APz);
    Vec3 FP_AP = FP - AP;

    doublereal h = std::fabs(FP_AP.dGet(3));
    doublereal L_APFP = std::sqrt(std::pow(FP_AP.dGet(1), 2) + std::pow(FP_AP.dGet(2), 2));

    Vec3 horizontal_unit(1.0, 0.0, 0.0);
    if (L_APFP > 1e-12) {
        horizontal_unit = Vec3(FP_AP.dGet(1)/L_APFP, FP_AP.dGet(2)/L_APFP, 0.0);
    }

    bool catenary_success = false;

    if (h > 1e-6 && L_APFP > 1e-6 && L > 1e-6) {
        try {
            doublereal L0_APFP = L - h;
            doublereal delta = L_APFP - L0_APFP;
            doublereal d = delta / h;
            doublereal l = L / h;

            if (d > 0 && d < (std::sqrt(l*l - 1) - (l - 1))) {
                doublereal p0 = 0.0;
                doublereal x1 = 0.0;
                doublereal x2 = 1.0e6;
                doublereal Ans_x = rtsafe(x1, x2, xacc, d, l, p0);
                doublereal a = Ans_x * h;
                doublereal segment_length = L / static_cast<doublereal>(NUM_SEGMENTS);

                for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
                    doublereal s = segment_length * static_cast<doublereal>(i + 1);
                    doublereal x_local, z_local;

                    if (p0 > 1e-6) {
                        doublereal s_contact = a * std::tan(p0);
                        if (s <= s_contact) {
                            x_local = s;
                            z_local = seabed_z - AP.dGet(3);
                        } else {
                            doublereal s_cat = s - s_contact;
                            doublereal beta = s_cat / a;
                            x_local = s_contact + a * myasinh(std::sinh(beta));
                            z_local = (seabed_z - AP.dGet(3)) + a * (std::cosh(beta) - 1.0);
                        }
                    } else {
                        doublereal theta_0 = -myasinh(L_APFP/a - 1.0/Ans_x);
                        doublereal beta = s/a + theta_0;
                        x_local = a * (std::sinh(beta) - std::sinh(theta_0));
                        z_local = a * (std::cosh(beta) - std::cosh(theta_0));
                    }

                    Vec3 initial_pos = AP + horizontal_unit*x_local + Vec3(0.0, 0.0, z_local);

                    initial_positions.push_back(initial_pos);
                }
                catenary_success = true;
            }
        } catch (...) {
            catenary_success = false;
        }
    }

    if (!catenary_success) {
        for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
            doublereal ratio = static_cast<doublereal>(i + 1) / static_cast<doublereal>(NUM_SEGMENTS);
            Vec3 linear_pos = AP + (FP - AP) * ratio;
            doublereal sag = 0.1 * L * ratio * (1.0 - ratio);
            Vec3 initial_pos = linear_pos + Vec3(0.0, 0.0, -sag);
            
            initial_positions.push_back(initial_pos);
        }
    }
}

// ========================================
// ========= MBDyn インターフェース =========
// ========================================

void ModuleCatenaryLM::Output(OutputHandler& OH) const
{
    if (bToBeOutput()) {
        if (OH.UseText(OutputHandler::LOADABLE)) {
            const Vec3& FP = g_pNode->GetXCurr();
            const Vec3& FV = g_pNode->GetVCurr();
            
            // フェアリーダーにかかる張力を最初のセグメント要素から取得
            Vec3 F_tension(0.0, 0.0, 0.0);
            Vec3 F_gravity_buoyancy(0.0, 0.0, 0.0);
            Vec3 F_seabed(0.0, 0.0, 0.0);
            Vec3 F_fluid(0.0, 0.0, 0.0);
            Vec3 F_total(0.0, 0.0, 0.0);

            doublereal current_length = 0.0;
            doublereal strain = 0.0;
            doublereal total_line_length = 0.0;

            if (!segment_elements.empty() && segment_elements[0] != nullptr) {
                // 最初のセグメント要素（フェアリーダー接続）から力を取得
                CatenarySegmentElement* first_segment = segment_elements[0];

                // セグメントの現在位置・速度取得
                const Vec3& x1 = first_segment->GetNode1()->GetXCurr();  // フェアリーダー
                const Vec3& x2 = first_segment->GetNode2()->GetXCurr();  // 最初の仮想ノード
                const Vec3& v1 = first_segment->GetNode1()->GetVCurr();
                const Vec3& v2 = first_segment->GetNode2()->GetVCurr();

                // ==== 軸力 =====
                Vec3 dx = x2 - x1;
                current_length = dx.Norm();

                if (current_length > 1e-12) {
                    Vec3 t = dx / current_length;
                    
                    doublereal L0 = first_segment->GetL0();
                    strain = (current_length - L0) / L0;
                    
                    doublereal F_elastic = (strain > 0.0) ? first_segment->GetEA() * strain : 0.0;
                    
                    Vec3 dv = v2 - v1;
                    doublereal v_axial = dv.Dot(t);
                    doublereal F_damping = first_segment-> GetCA() * v_axial;
                    
                    doublereal F_axial_magnitude = F_elastic + F_damping;
                    F_tension = t * F_axial_magnitude;
                }

                // ===== 重力・浮力　=====
                doublereal volume = (M_PI / 4.0) * first_segment->GetLineDiameter() * first_segment->GetLineDiameter() * first_segment->GetL0();
                doublereal weight = first_segment->GetSegmentMass() * first_segment->GetGravity();
                doublereal buoyancy = first_segment->GetRhoWater() * volume * first_segment->GetGravity();
                F_gravity_buoyancy = Vec3(0.0, 0.0, -(weight - buoyancy)) * 0.5;

                // ===== 海底反力 =====
                doublereal z = x1.dGet(3);
                doublereal penetration = first_segment->GetSeabedZ() - z;
                if (penetration > 0.0) {
                    doublereal Fz = first_segment->GetKSeabed()*penetration - first_segment->GetCSeabed() * v1.dGet(3);
                    F_seabed = Vec3(0.0, 0.0, std::max(0.0, Fz));
                }

                // ===== 流体力 ======
                doublereal drag_coeff = 1.2;
                doublereal drag_area = first_segment->GetLineDiameter() * first_segment->GetL0();
                doublereal vz = v1.dGet(3);
                if (std::abs(vz) > 1e-12) {
                    doublereal F_drag = -0.5 * first_segment->GetRhoWater() * drag_coeff * drag_area * vz * std::abs(vz);
                    F_fluid = Vec3(0.0, 0.0, F_drag);
                }

                // ====== 総合力 =======
                // フェアリーダーには軸力の反作用が働く TODO
                F_total = -F_tension + F_gravity_buoyancy + F_seabed + F_fluid;
            }

            // 全セグメントの現在長を合計
            for (unsigned int i = 0; i < segment_elements.size(); ++i) {
                if (segment_elements[i] != nullptr) {
                    const Vec3& seg_x1 = segment_elements[i]->GetNode1()->GetXCurr();
                    const Vec3& seg_x2 = segment_elements[i]->GetNode2()->GetXCurr();
                    total_line_length += (seg_x2 - seg_x1).Norm();
                }
            }

            // 係留索の形状解析
            doublereal max_z = FP.dGet(3);
            doublereal min_z = APz;
            doublereal horizontal_span = std::sqrt(std::pow(FP.dGet(1) - APx, 2) + std::pow(FP.dGet(2) - APy, 2));
            
            for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
                if (virtual_nodes[i] != nullptr) {
                    doublereal node_z = virtual_nodes[i]->GetXCurr().dGet(3);
                    if (node_z < min_z) {
                        min_z = node_z;
                    }
                }
            }    
            doublereal sag = max_z - min_z;

            // 海底接触
            unsigned int nodes_on_seabed = 0;
            doublereal contact_length = 0.0;
            
            for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
                if (virtual_nodes[i] != nullptr) {
                    doublereal node_z = virtual_nodes[i]->GetXCurr().dGet(3);
                    if (node_z <= seabed_z + 0.01) {  // 僅かな許容誤差
                        nodes_on_seabed++;
                    }
                }
            }

            // 接触長の概算
            if (nodes_on_seabed > 0) {
                contact_length = (static_cast<doublereal>(nodes_on_seabed) / static_cast<doublereal>(NUM_SEGMENTS)) * L;
            }

            // === 動的特性解析 ===
            doublereal kinetic_energy = 0.0;
            doublereal potential_energy = 0.0;
            
            // フェアリーダーの運動エネルギー
            doublereal fairlead_mass = w / static_cast<doublereal>(NUM_SEGMENTS);  // 概算
            kinetic_energy += 0.5 * fairlead_mass * FV.Dot(FV);
            
            // 仮想ノードの運動・位置エネルギー
            for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
                if (virtual_nodes[i] != nullptr) {
                    const Vec3& node_pos = virtual_nodes[i]->GetXCurr();
                    const Vec3& node_vel = virtual_nodes[i]->GetVCurr();
                    
                    doublereal node_mass = rho_line * L / static_cast<doublereal>(NUM_SEGMENTS);
                    
                    // 運動エネルギー
                    kinetic_energy += 0.5 * node_mass * node_vel.Dot(node_vel);
                    
                    // 重力ポテンシャルエネルギー（海底を基準）
                    potential_energy += node_mass * g_gravity * (node_pos.dGet(3) - seabed_z);
                }
            }
            
            doublereal total_energy = kinetic_energy + potential_energy;

            // === 出力 ===
            OH.Loadable() << GetLabel()
                // フェアリーダー位置 (1-3)
                << " " << FP.dGet(1)              // X位置
                << " " << FP.dGet(2)              // Y位置  
                << " " << FP.dGet(3)              // Z位置
                
                // フェアリーダー速度 (4-6)
                << " " << FV.dGet(1)              // X速度
                << " " << FV.dGet(2)              // Y速度
                << " " << FV.dGet(3)              // Z速度
                
                // 張力成分 (7-10)
                << " " << F_tension.dGet(1)       // 張力X
                << " " << F_tension.dGet(2)       // 張力Y
                << " " << F_tension.dGet(3)       // 張力Z
                << " " << F_tension.Norm()        // 張力大きさ
                
                // 総合力成分 (11-14)
                << " " << F_total.dGet(1)         // 総合力X
                << " " << F_total.dGet(2)         // 総合力Y
                << " " << F_total.dGet(3)         // 総合力Z
                << " " << F_total.Norm()          // 総合力大きさ
                
                // 幾何学的特性 (15-20)
                << " " << current_length          // 最初のセグメント長
                << " " << strain                  // 最初のセグメントひずみ
                << " " << total_line_length       // 総現在長
                << " " << horizontal_span         // 水平スパン
                << " " << sag                     // サグ（たわみ）
                << " " << contact_length          // 海底接触長
                
                // 力成分詳細 (21-32)
                << " " << F_gravity_buoyancy.dGet(1) << " " << F_gravity_buoyancy.dGet(2) << " " << F_gravity_buoyancy.dGet(3)  // 重力浮力
                << " " << F_seabed.dGet(1) << " " << F_seabed.dGet(2) << " " << F_seabed.dGet(3)                              // 海底反力
                << " " << F_fluid.dGet(1) << " " << F_fluid.dGet(2) << " " << F_fluid.dGet(3)                                // 流体力
                << " " << FSF.dGet()              // Force Scale Factor
                
                // エネルギー・統計 (34-37)
                << " " << kinetic_energy          // 運動エネルギー
                << " " << potential_energy        // ポテンシャルエネルギー
                << " " << total_energy            // 総エネルギー
                << " " << nodes_on_seabed         // 海底接触ノード数
                
                << std::endl;

        }
    }
}

void ModuleCatenaryLM::WorkSpaceDim(integer* piNumRows, integer* piNumCols) const
{
    // メイン要素は力を生成せず，セグメント要素で処理する
    *piNumRows = 0;
    *piNumCols = 0;
}

VariableSubMatrixHandler& ModuleCatenaryLM::AssJac(
    VariableSubMatrixHandler& WorkMat,
    doublereal dCoef,
    const VectorHandler& XCurr,
    const VectorHandler& XPrimeCurr)
{
    // メイン要素ではなくセグメント要素で処理
    WorkMat.SetNullMatrix();
    return WorkMat;
}

SubVectorHandler& ModuleCatenaryLM::AssRes(
    SubVectorHandler& WorkVec,
    doublereal dCoef,
    const VectorHandler& XCurr,
    const VectorHandler& XPrimeCurr)
{
    // メイン要素ではなくセグメント要素で処理
    WorkVec.ResizeReset(0);
    return WorkVec;
}

unsigned int ModuleCatenaryLM::iGetNumPrivData(void) const
{
    return 0;
}

int ModuleCatenaryLM::iGetNumConnectedNodes(void) const
{
    return 1 + virtual_nodes.size(); // フェアリーダー + 仮想ノード
}

void ModuleCatenaryLM::GetConnectedNodes(std::vector<const Node *>& connectedNodes) const
{
    connectedNodes.resize(1 + virtual_nodes.size());
    connectedNodes[0] = g_pNode; // フェアリーダーノード

    for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
        connectedNodes[i+1] = virtual_nodes[i];
    }
}

void ModuleCatenaryLM::SetValue(
    DataManager *pDM, 
    VectorHandler& X, 
    VectorHandler& XP, 
    SimulationEntity::Hints *ph)
{
    // 初期位置が設定されていない場合は設定
    if (initial_positions.empty()) {
        InitializeVirtualNodesFromCatenary();
    }

    // カテナリー理論による初期位置を MBDyn の状態ベクトルに反映
    for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
        if (i < initial_positions.size()) {
            const Vec3& initial_pos = initial_positions[i];
        
            integer first_pos_idx = virtual_nodes[i]->iGetFirstPositionIndex();
            X.PutCoef(first_pos_idx + 1, initial_pos.dGet(1));
            X.PutCoef(first_pos_idx + 2, initial_pos.dGet(2));
            X.PutCoef(first_pos_idx + 3, initial_pos.dGet(3));
        
            // 初期速度はゼロ
            XP.PutCoef(first_pos_idx + 1, 0.0);
            XP.PutCoef(first_pos_idx + 2, 0.0);
            XP.PutCoef(first_pos_idx + 3, 0.0);
        }
    }
}

std::ostream& ModuleCatenaryLM::Restart(std::ostream& out) const
{
    return out << "# ModuleCatenaryLM: not implemented" << std::endl;
}

unsigned int ModuleCatenaryLM::iGetInitialNumDof(void) const
{
    return 0;
}

void ModuleCatenaryLM::InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const
{
    *piNumRows = 0;
    *piNumCols = 0;
}

VariableSubMatrixHandler& ModuleCatenaryLM::InitialAssJac(
    VariableSubMatrixHandler& WorkMat,
    const VectorHandler& XCurr)
{
    ASSERT(0);
    WorkMat.SetNullMatrix();
    return WorkMat;
}

SubVectorHandler& ModuleCatenaryLM::InitialAssRes(
    SubVectorHandler& WorkVec,
    const VectorHandler& XCurr)
{
    ASSERT(0);
    WorkVec.ResizeReset(0);
    return WorkVec;
}

// =================================
// ========= モジュール登録 =========
// =================================

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

// =============================
// ======= セグメント要素 =======
// =============================

// コンストラクタ
CatenarySegmentElement::CatenarySegmentElement(
    unsigned uLabel, const DofOwner* pDO,
    const StructNode* n1, const StructNode* n2,
    doublereal ea, doublereal ca, doublereal l0, 
    doublereal mass, doublereal diameter,
    doublereal rho_l, doublereal rho_w, doublereal gravity,
    doublereal sb_z, doublereal k_sb, doublereal c_sb,
    DriveCaller* pFSF)
    : Elem(uLabel, flag(0)),
      UserDefinedElem(uLabel, pDO),
      node1(n1), node2(n2),
      EA(ea), CA(ca), L0(l0), segment_mass(mass),
      line_diameter(diameter), rho_line(rho_l), rho_water(rho_w),
      g_gravity(gravity), seabed_z(sb_z), K_seabed(k_sb), C_seabed(c_sb)
{
    FSF.Set(pFSF ? pFSF : new OneDriveCaller);
}

CatenarySegmentElement::~CatenarySegmentElement(void)
{
    NO_OP;
}

Vec3 CatenarySegmentElement::ComputeAxialForce(
    const Vec3& x1, const Vec3& x2, 
    const Vec3& v1, const Vec3& v2) const
{
    Vec3 dx = x2 - x1;
    doublereal l_current = dx.Norm();
    
    if (l_current < 1e-12) {
        return Vec3(0.0, 0.0, 0.0);
    }
    
    Vec3 t = dx / l_current;
    doublereal strain = (l_current - L0) / L0;
    doublereal F_elastic = (strain > 0.0) ? EA * strain : 0.0;
    
    Vec3 dv = v2 - v1;
    doublereal v_axial = dv.Dot(t);
    doublereal F_damping = CA * v_axial;
    
    doublereal F_total = F_elastic + F_damping;
    
    return t * F_total;
}

Vec3 CatenarySegmentElement::ComputeGravityBuoyancy() const
{
    doublereal volume = (M_PI / 4.0) * line_diameter * line_diameter * L0;
    doublereal weight = segment_mass * g_gravity;
    doublereal buoyancy = rho_water * volume * g_gravity;
    
    return Vec3(0.0, 0.0, -(weight - buoyancy));
}

Vec3 CatenarySegmentElement::ComputeSeabedForce(const Vec3& position, const Vec3& velocity) const
{
    Vec3 F_seabed(0.0, 0.0, 0.0);
    
    doublereal z = position.dGet(3);
    doublereal penetration = seabed_z - z;
    
    if (penetration > 0.0) {
        doublereal Fz = K_seabed * penetration - C_seabed * velocity.dGet(3);
        F_seabed = Vec3(0.0, 0.0, std::max(0.0, Fz));
    }
    
    return F_seabed;
}

Vec3 CatenarySegmentElement::ComputeFluidForce(const Vec3& position, const Vec3& velocity) const
{
    doublereal added_mass_coeff = 1.0;
    doublereal added_mass = added_mass_coeff * rho_water * (M_PI / 4.0) * line_diameter * line_diameter * L0;
    
    doublereal drag_coeff = 1.2;
    doublereal drag_area = line_diameter * L0;
    
    Vec3 F_fluid(0.0, 0.0, 0.0);
    
    doublereal vz = velocity.dGet(3);
    if (std::abs(vz) > 1e-12) {
        doublereal F_drag = -0.5 * rho_water * drag_coeff * drag_area * vz * std::abs(vz);
        F_fluid += Vec3(0.0, 0.0, F_drag);
    }
    
    return F_fluid;
}

// 剛性行列計算
void CatenarySegmentElement::ComputeStiffnessMatrix(
    const Vec3& x1, const Vec3& x2,
    Mat3x3& K11, Mat3x3& K12, 
    Mat3x3& K21, Mat3x3& K22)
{
    Vec3 dx = x2 - x1;
    doublereal l = dx.Norm();
    
    if (l < 1e-12) {
        K11 = K12 = K21 = K22 = Zero3x3;
        return;
    }
    
    Vec3 t = dx / l;
    doublereal strain = (l - L0) / L0;
    
    Mat3x3 tt = Mat3x3(MatCross, t) * Mat3x3(MatCross, t);
    Mat3x3 I_tt = Eye3 - tt;
    
    doublereal k_axial = EA / L0;
    doublereal k_trans = 0.0;
    
    if (strain > 0.0) {
        k_trans = EA * strain / l;
    }
    
    K11 = tt * k_axial + I_tt * k_trans;
    K12 = -K11;
    K21 = -K11;
    K22 = K11;
}

void CatenarySegmentElement::ComputeDampingMatrix(
    const Vec3& x1, const Vec3& x2,
    Mat3x3& C11, Mat3x3& C12, 
    Mat3x3& C21, Mat3x3& C22)
{
    Vec3 dx = x2 - x1;
    doublereal l = dx.Norm();
    
    if (l < 1e-12) {
        C11 = C12 = C21 = C22 = Zero3x3;
        return;
    }
    
    Vec3 t = dx / l;  // 単位ベクトル
    Mat3x3 tt = Mat3x3(MatCross, t) * Mat3x3(MatCross, t);  // t⊗t
    
    doublereal c_axial = CA;
    
    C11 = tt * c_axial;
    C12 = -C11;
    C21 = -C11;
    C22 = C11;
}

// 海底剛性・減衰
Mat3x3 CatenarySegmentElement::ComputeSeabedStiffness(const Vec3& position)
{
    doublereal z = position.dGet(3);
    doublereal penetration = seabed_z - z;
    
    if (penetration > 0.0) {
        Mat3x3 K_sb = Zero3x3;
        K_sb(3, 3) = K_seabed;
        return K_sb;
    }
    
    return Zero3x3;
}

Mat3x3 CatenarySegmentElement::ComputeSeabedDamping(const Vec3& position)
{
    doublereal z = position.dGet(3);
    doublereal penetration = seabed_z - z;
    
    if (penetration > 0.0) {
        Mat3x3 C_sb = Zero3x3;
        C_sb(3, 3) = C_seabed;
        return C_sb;
    }
    
    return Zero3x3;
}

// MBDynインターフェース
void CatenarySegmentElement::Output(OutputHandler& OH) const
{
    if (bToBeOutput()) {
        if (OH.UseText(OutputHandler::LOADABLE)) {
            const Vec3& x1 = node1->GetXCurr();
            const Vec3& x2 = node2->GetXCurr();
            const Vec3& v1 = node1->GetVCurr();
            const Vec3& v2 = node2->GetVCurr();
            
            Vec3 F_axial = ComputeAxialForce(x1, x2, v1, v2);
            doublereal current_length = (x2 - x1).Norm();
            doublereal strain = (current_length - L0) / L0;
            
            OH.Loadable() << GetLabel()
                << " " << current_length    // 現在長
                << " " << strain           // ひずみ
                << " " << F_axial.Norm()   // 軸力大きさ
                << " " << F_axial.dGet(1)  // 軸力X
                << " " << F_axial.dGet(2)  // 軸力Y
                << " " << F_axial.dGet(3)  // 軸力Z
                << std::endl;
        }
    }
}

void CatenarySegmentElement::WorkSpaceDim(integer* piNumRows, integer* piNumCols) const
{
    *piNumRows = 12;  // 2ノード × 6DOF
    *piNumCols = 12;
}

SubVectorHandler& CatenarySegmentElement::AssRes(
    SubVectorHandler& WorkVec,
    doublereal dCoef,
    const VectorHandler& XCurr,
    const VectorHandler& XPrimeCurr)
{
    WorkVec.ResizeReset(12);
    
    // ノードインデックス設定
    integer idx1 = node1->iGetFirstMomentumIndex();
    integer idx2 = node2->iGetFirstMomentumIndex();
    
    for (int i = 1; i <= 6; i++) {
        WorkVec.PutRowIndex(i, idx1 + i);
        WorkVec.PutRowIndex(i + 6, idx2 + i);
    }
    
    // 現在の位置・速度取得
    const Vec3& x1 = node1->GetXCurr();
    const Vec3& x2 = node2->GetXCurr();
    const Vec3& v1 = node1->GetVCurr();
    const Vec3& v2 = node2->GetVCurr();
    
    // Force Scale Factor
    doublereal dFSF = FSF.dGet();
    
    // === ランプドマス法の力計算 ===
    
    // 1. 軸力（バネ・ダンパー）
    Vec3 F_axial = ComputeAxialForce(x1, x2, v1, v2);
    
    // 2. 重力・浮力（各ノードに半分ずつ配分）
    Vec3 F_gravity_buoyancy = ComputeGravityBuoyancy();
    
    // 3. 海底反力
    Vec3 F_seabed1 = ComputeSeabedForce(x1, v1);
    Vec3 F_seabed2 = ComputeSeabedForce(x2, v2);
    
    // 4. 流体力
    Vec3 F_fluid1 = ComputeFluidForce(x1, v1);
    Vec3 F_fluid2 = ComputeFluidForce(x2, v2);
    
    // === 残差に力を追加 ===
    // Node1: -軸力 + 重力浮力の半分 + 海底力 + 流体力
    Vec3 F1 = (-F_axial + F_gravity_buoyancy * 0.5 + F_seabed1 + F_fluid1) * dFSF;
    WorkVec.Add(1, F1);
    
    // Node2: +軸力 + 重力浮力の半分 + 海底力 + 流体力  
    Vec3 F2 = (F_axial + F_gravity_buoyancy * 0.5 + F_seabed2 + F_fluid2) * dFSF;
    WorkVec.Add(7, F2);
    
    // モーメントは考慮しない（4-6, 10-12は0のまま）
    
    return WorkVec;
}

VariableSubMatrixHandler& CatenarySegmentElement::AssJac(
    VariableSubMatrixHandler& WorkMat,
    doublereal dCoef,
    const VectorHandler& XCurr,
    const VectorHandler& XPrimeCurr)
{
    WorkMat.SetNullMatrix();
    FullSubMatrixHandler& WM = WorkMat.SetFull();
    WM.ResizeReset(12, 12);
    
    // ノードインデックス設定
    integer idx1_pos = node1->iGetFirstPositionIndex();
    integer idx2_pos = node2->iGetFirstPositionIndex();
    integer idx1_mom = node1->iGetFirstMomentumIndex();
    integer idx2_mom = node2->iGetFirstMomentumIndex();
    
    for (int i = 1; i <= 6; i++) {
        WM.PutRowIndex(i, idx1_mom + i);
        WM.PutRowIndex(i + 6, idx2_mom + i);
        WM.PutColIndex(i, idx1_pos + i);
        WM.PutColIndex(i + 6, idx2_pos + i);
    }
    
    // 現在の位置・速度
    const Vec3& x1 = node1->GetXCurr();
    const Vec3& x2 = node2->GetXCurr();
    
    // Force Scale Factor
    doublereal dFSF = FSF.dGet();
    
    // 軸力の位置による偏微分（剛性行列）
    Mat3x3 K11, K12, K21, K22;
    ComputeStiffnessMatrix(x1, x2, K11, K12, K21, K22);
    
    // 軸力の速度による偏微分（減衰行列）
    Mat3x3 C11, C12, C21, C22;
    ComputeDampingMatrix(x1, x2, C11, C12, C21, C22);
    
    // 海底反力の偏微分
    Mat3x3 K_seabed1 = ComputeSeabedStiffness(x1);
    Mat3x3 K_seabed2 = ComputeSeabedStiffness(x2);
    Mat3x3 C_seabed1 = ComputeSeabedDamping(x1);
    Mat3x3 C_seabed2 = ComputeSeabedDamping(x2);
    
    // ヤコビアン = K + dCoef * C（力方程式の並進成分のみ）
    WM.Add(1, 1, (K11 + K_seabed1 + (C11 + C_seabed1) * dCoef) * dFSF);
    WM.Add(1, 7, (K12 + C12 * dCoef) * dFSF);
    WM.Add(7, 1, (K21 + C21 * dCoef) * dFSF);
    WM.Add(7, 7, (K22 + K_seabed2 + (C22 + C_seabed2) * dCoef) * dFSF);
    
    return WorkMat;
}

unsigned int CatenarySegmentElement::iGetNumPrivData(void) const
{
    return 0;
}

int CatenarySegmentElement::iGetNumConnectedNodes(void) const
{
    return 2;
}

void CatenarySegmentElement::GetConnectedNodes(std::vector<const Node*>& connectedNodes) const
{
    connectedNodes.resize(2);
    connectedNodes[0] = node1;
    connectedNodes[1] = node2;
}

void CatenarySegmentElement::SetValue(
    DataManager *pDM, 
    VectorHandler& X, 
    VectorHandler& XP,
    SimulationEntity::Hints *ph)
{
    NO_OP;
}

std::ostream& CatenarySegmentElement::Restart(std::ostream& out) const
{
    return out << "# CatenarySegmentElement: restart not implemented" << std::endl;
}

unsigned int CatenarySegmentElement::iGetInitialNumDof(void) const
{
    return 0;
}

void CatenarySegmentElement::InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const
{
    *piNumRows = 0;
    *piNumCols = 0;
}

VariableSubMatrixHandler& CatenarySegmentElement::InitialAssJac(
    VariableSubMatrixHandler& WorkMat,
    const VectorHandler& XCurr)
{
    ASSERT(0);
    WorkMat.SetNullMatrix();
    return WorkMat;
}

SubVectorHandler& CatenarySegmentElement::InitialAssRes(
    SubVectorHandler& WorkVec,
    const VectorHandler& XCurr)
{
    ASSERT(0);
    WorkVec.ResizeReset(0);
    return WorkVec;
}
