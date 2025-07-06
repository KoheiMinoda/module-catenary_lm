
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
#include "module-catenary_lm.h" // 自前のヘッダ

// ========= 仮想ノード構造体 =========
struct VirtualNode {
    Vec3 position;      // 位置
    Vec3 velocity;      // 速度
    Vec3 acceleration;  // 加速度
    doublereal mass;    // 質量
    bool active;        // アクティブフラグ
    
    VirtualNode() : 
        position(0.0, 0.0, 0.0), 
        velocity(0.0, 0.0, 0.0),
        acceleration(0.0, 0.0, 0.0),
        mass(0.0), 
        active(true) 
    {}
    
    VirtualNode(const Vec3& pos, doublereal m) : 
        position(pos), 
        velocity(0.0, 0.0, 0.0),
        acceleration(0.0, 0.0, 0.0),
        mass(m), 
        active(true) 
    {}
};

class ModuleCatenaryLM
    : virtual public Elem, public UserDefinedElem {
    public:
        // ----コンストラクタとデストラクタ----
        ModuleCatenaryLM(unsigned uLabel, const DofOwner *pDO, DataManager* pDM, MBDynParser& HP);
        virtual ~ModuleCatenaryLM(void);

        // ----シミュレーション関連メソッド----
        virtual void Output(OutputHandler& OH) const;
        virtual void WorkSpaceDim(integer* piNumRows, integer* piNumCols) const;

        VariableSubMatrixHandler& 
        AssJac(VariableSubMatrixHandler& WorkMat,
            doublereal dCoef, 
            const VectorHandler& XCurr,
            const VectorHandler& XPrimeCurr);

        SubVectorHandler& 
        AssRes(SubVectorHandler& WorkVec,
            doublereal dCoef,
            const VectorHandler& XCurr, 
            const VectorHandler& XPrimeCurr);

        // ----データアクセスメソッド----
        unsigned int iGetNumPrivData(void) const;
        int iGetNumConnectedNodes(void) const;
        void GetConnectedNodes(std::vector<const Node *>& connectedNodes) const;
        void SetValue(DataManager *pDM, VectorHandler& X, VectorHandler& XP,
            SimulationEntity::Hints *ph);
        std::ostream& Restart(std::ostream& out) const;

        // 初期化関連メソッド
        virtual unsigned int iGetInitialNumDof(void) const;
        virtual void InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const;
        VariableSubMatrixHandler&
        InitialAssJac(VariableSubMatrixHandler& WorkMat, 
            const VectorHandler& XCurr);
        SubVectorHandler& 
        InitialAssRes(SubVectorHandler& WorkVec, const VectorHandler& XCurr);

    private:
        const StructNode* g_pNode;              // フェアリーダーを表す MBDyn の構造ノード
        
        // 基本パラメータ（元のmodule-catenary.ccから継承）
        double APx, APy, APz;                   // アンカーのグローバル x,y,z 座標
        double L;                               // ライン全長
        double w;                               // チェーンの単重
        double xacc;                            // rtsafe精度
        
        // ランプドマス法パラメータ
        doublereal EA;                          // 軸剛性
        doublereal CA;                          // 軸減衰
        doublereal rho_line;                    // 線密度
        doublereal line_diameter;               // 線径
        doublereal g_gravity;                   // 重力加速度
        
        // 海底パラメータ
        doublereal seabed_z;                    // 海底z座標
        doublereal K_seabed;                    // 海底剛性
        doublereal C_seabed;                    // 海底減衰
        
        // 時間関連
        doublereal simulation_time;             // シミュレーション時間
        doublereal prev_time;                   // 前のタイムステップの時間
        doublereal ramp_time;                   // ランプアップ時間
        
        DriveOwner FSF;                         // Force Scale Factor
        
        // 仮想ノード管理
        std::vector<VirtualNode> virtual_nodes; // 内部仮想ノード
        static const unsigned int NUM_SEGMENTS = 20; // セグメント数
        
        // カテナリー理論関数
        static double myasinh(double X);
        static double myacosh(double X);
        static void funcd(double x, double xacc, double &f, double &df, double d, double l, double &p0);
        static double rtsafe(double x1, double x2, double xacc, double d, double l, double &p0);
        
        // ランプドマス法関数
        void InitializeVirtualNodesFromCatenary();
        void UpdateVirtualNodes(doublereal dt);
        doublereal GetRampFactor(doublereal current_time) const;
        void ComputeLumpedMassForces(Vec3& F_total, Vec3& M_total);
        Vec3 ComputeAxialForce(const Vec3& pos1, const Vec3& pos2, const Vec3& vel1, const Vec3& vel2, doublereal L0) const;
        Vec3 ComputeSeabedForce(const Vec3& position, const Vec3& velocity) const;
};

// ========= コンストラクタ =========
ModuleCatenaryLM::ModuleCatenaryLM(
    unsigned uLabel,
    const DofOwner *pDO, 
    DataManager* pDM, 
    MBDynParser& HP
)
    : Elem(uLabel, flag(0)), UserDefinedElem(uLabel, pDO), 
        g_pNode(0), 
        APx(0), APy(0), APz(0),
        L(0), w(0), xacc(1e-6),
        EA(3.842e8), CA(0.0), 
        rho_line(77.71), line_diameter(0.09017),
        g_gravity(9.80665),
        seabed_z(-320.0), K_seabed(1.0e5), C_seabed(1.0e3),
        simulation_time(0.0), prev_time(0.0), ramp_time(10.0)
{
    // help
    if (HP.IsKeyWord("help")) {
        silent_cout(
            "\n"
            "Module: 	ModuleCatenaryLM (Lumped Mass Method)\n"
            "Usage:      catenary_lm, fairlead_node_label, \n"
            "                LineLength, total_length,\n"
            "                LineWeight, unit_weight,\n"
            "                Xacc, rtsafe_accuracy,\n"
            "                APx, APy, APz,\n"
            "              [ EA, axial_stiffness, ]\n"
            "              [ CA, axial_damping, ]\n"
            "              [ rho_line, line_density, ]\n"
            "              [ line_diameter, diameter, ]\n"
            "              [ gravity, g_acceleration, ]\n"
            "              [ seabed, z_coordinate, k_stiffness, c_damping, ]\n"
            "              [ ramp_time, ramp_duration, ]\n"
            "              [ force scale factor, (DriveCaller), ]\n"
            "              [ output, (FLAG) ] ;\n"
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
    if (HP.IsKeyWord("EA")) {
        EA = HP.GetReal();
    }
    if (HP.IsKeyWord("CA")) {
        CA = HP.GetReal();
    }
    if (HP.IsKeyWord("rho_line")) {
        rho_line = HP.GetReal();
    }
    if (HP.IsKeyWord("line_diameter")) {
        line_diameter = HP.GetReal();
    }
    if (HP.IsKeyWord("gravity")) {
        g_gravity = HP.GetReal();
    }
    if (HP.IsKeyWord("seabed")) {
        seabed_z = HP.GetReal();
        K_seabed = HP.GetReal();
        C_seabed = HP.GetReal();
    }
    if (HP.IsKeyWord("ramp_time")) {
        ramp_time = HP.GetReal();
    }

    // Force Scale Factor
    if (HP.IsKeyWord("Force" "scale" "factor")) {
        FSF.Set(HP.GetDriveCaller());
    } else {
        FSF.Set(new OneDriveCaller);
    }

    SetOutputFlag(pDM->fReadOutput(HP, Elem::LOADABLE));

    // 仮想ノード初期化
    virtual_nodes.resize(NUM_SEGMENTS - 1); // N-1個の内部ノード
    doublereal segment_mass = (rho_line * L) / static_cast<doublereal>(NUM_SEGMENTS);
    
    for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
        virtual_nodes[i].mass = segment_mass;
        virtual_nodes[i].active = true;
    }

    // カテナリー理論による初期位置設定
    InitializeVirtualNodesFromCatenary();

    pDM->GetLogFile() << "catenary_lm: "
        << uLabel << " "
        << "Segments: " << NUM_SEGMENTS << " "
        << "VirtualNodes: " << virtual_nodes.size() << " "
        << "EA: " << EA << " "
        << "rho_line: " << rho_line << " "
        << std::endl;
}

ModuleCatenaryLM::~ModuleCatenaryLM(void)
{
    // destroy private data
    NO_OP;
}

// ========= カテナリー理論関数（元コードから継承） =========
double ModuleCatenaryLM::myasinh(double X) 
{
    return std::log(X + std::sqrt(X * X + 1));
}

double ModuleCatenaryLM::myacosh(double X) 
{
    return std::log(X + std::sqrt(X + 1) * std::sqrt(X - 1));
}

void ModuleCatenaryLM::funcd(double x, double xacc, double& f, double& df, double d, double l, double& p0)
{
    int max = 1000;
    double f1, df1;

    // 係留索が完全にたるんでいて，水平張力 0
    if(x == 0.0) {
        f = -d;
        df = 0.0;
        p0 = 0.0;
    }
    // 水平張力あり
    else if(x > 0.0) {
        // 全長が垂直距離以下という物理的にあり得ない状況
        if(l <= 0.0) {
            double X_1 = 1.0/x + 1.0;
            f = x*myacosh(X_1) - std::sqrt(1.0 + 2.0*x) + 1.0 - d;
            df = myacosh(X_1) - 1.0/std::sqrt(1.0 + 2.0*x) - 1.0/(x*std::sqrt(std::pow(X_1, 2.0) - 1.0));
            p0 = 0.0;
        } else {
            // 海底に接する可能性のある複雑なケース
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
                // 単純なカテナリー
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

    if(fl == 0.0) {
        p0 = p1;
        return x1;
    }
    if(fh == 0.0) {
        p0 = p2;
        return x2;
    }

    if(fl < 0.0) {
        xl = x1;
        xh = x2;
    } else {
        xh = x1;
        xl = x2;
    }

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

        if(f < 0.0) {
            xl = rts;
        } else {
            xh = rts;
        }
    }

    std::cout << "ERROR (Bisection method)" << std::endl;
    throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
}

// ========= カテナリー理論による仮想ノード初期位置設定 =========
void ModuleCatenaryLM::InitializeVirtualNodesFromCatenary()
{
    if (g_pNode == nullptr || virtual_nodes.empty()) {
        return;
    }

    const Vec3& FP = g_pNode->GetXCurr();
    Vec3 AP(APx, APy, APz);
    
    // アンカーを原点とする座標変換
    Vec3 FP_AP = FP - AP;
    
    // アンカー・フェアリーダー間の鉛直距離と水平距離
    doublereal h = std::fabs(FP_AP.dGet(3));
    doublereal L_APFP = std::sqrt(std::pow(FP_AP.dGet(1), 2) + std::pow(FP_AP.dGet(2), 2));
    
    // 水平方向の単位ベクトル
    Vec3 horizontal_unit(1.0, 0.0, 0.0);
    if (L_APFP > 1e-12) {
        horizontal_unit = Vec3(FP_AP.dGet(1)/L_APFP, FP_AP.dGet(2)/L_APFP, 0.0);
    }

    bool catenary_success = false;

    // カテナリー理論による計算
    if (h > 1e-6 && L_APFP > 1e-6 && L > 1e-6) {
        try {
            // カテナリー計算パラメータ
            doublereal L0_APFP = L - h;
            doublereal delta = L_APFP - L0_APFP;
            doublereal d = delta / h;
            doublereal l = L / h;
            
            if (d > 0 && d < (std::sqrt(l*l - 1) - (l - 1))) {
                doublereal p0 = 0.0;
                doublereal x1 = 0.0;
                doublereal x2 = 1.0e6;
                
                // 水平張力パラメータを求解
                doublereal Ans_x = rtsafe(x1, x2, xacc, d, l, p0);
                
                // カテナリーパラメータ
                doublereal a = Ans_x * h; // H / w = Ans_x * w * h / w
                
                // 各仮想ノードの位置計算
                doublereal segment_length = L / static_cast<doublereal>(NUM_SEGMENTS);
                
                for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
                    // アンカーからの弧長距離
                    doublereal s = segment_length * static_cast<doublereal>(i + 1);
                    
                    // カテナリー方程式による座標計算
                    doublereal x_local, z_local;
                    
                    if (p0 > 1e-6) {
                        // 海底接触ありの場合（簡略化）
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
                        // 海底接触なしの場合
                        doublereal theta_0 = -myasinh(L_APFP/a - 1.0/Ans_x);
                        doublereal beta = s/a + theta_0;
                        
                        x_local = a * (std::sinh(beta) - std::sinh(theta_0));
                        z_local = a * (std::cosh(beta) - std::cosh(theta_0));
                    }
                    
                    // グローバル座標への変換
                    virtual_nodes[i].position = AP + horizontal_unit * x_local + Vec3(0.0, 0.0, z_local);
                    virtual_nodes[i].velocity = Vec3(0.0, 0.0, 0.0);
                    virtual_nodes[i].active = true;
                }
                
                catenary_success = true;
            }
        } catch (...) {
            catenary_success = false;
        }
    }

    // フォールバック：線形補間
    if (!catenary_success) {
        for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
            doublereal ratio = static_cast<doublereal>(i + 1) / static_cast<doublereal>(NUM_SEGMENTS);
            virtual_nodes[i].position = AP + (FP - AP) * ratio;
            
            // 軽微な垂れ下がりを追加
            doublereal sag = 0.1 * L * ratio * (1.0 - ratio);
            virtual_nodes[i].position += Vec3(0.0, 0.0, -sag);
            
            virtual_nodes[i].velocity = Vec3(0.0, 0.0, 0.0);
            virtual_nodes[i].active = true;
        }
    }
}

// ========= ランプファクター計算 =========
doublereal ModuleCatenaryLM::GetRampFactor(doublereal current_time) const
{
    if (current_time <= 0.0) {
        return 0.0;
    } else if (current_time >= ramp_time) {
        return 1.0;
    } else {
        return current_time / ramp_time;
    }
}

// ========= 軸力計算 =========
Vec3 ModuleCatenaryLM::ComputeAxialForce(const Vec3& pos1, const Vec3& pos2, const Vec3& vel1, const Vec3& vel2, doublereal L0) const
{
    Vec3 dx = pos2 - pos1;
    doublereal l_current = dx.Norm();
    
    if (l_current < 1e-12) {
        return Vec3(0.0, 0.0, 0.0);
    }
    
    Vec3 t = dx / l_current;
    
    // ひずみ計算
    doublereal strain = (l_current - L0) / L0;
    
    // 弾性力（引張のみ）
    doublereal Fel = (strain > 0.0) ? EA * strain : 0.0;
    
    // 減衰力
    Vec3 dv = vel2 - vel1;
    doublereal vrel = dv.Dot(t);
    doublereal Fd = CA * vrel;
    
    doublereal Fax = Fel + Fd;
    
    return t * Fax;
}

// ========= 海底反力計算 =========
Vec3 ModuleCatenaryLM::ComputeSeabedForce(const Vec3& position, const Vec3& velocity) const
{
    Vec3 F_seabed(0.0, 0.0, 0.0);
    
    doublereal z = position.dGet(3);
    doublereal penetration = seabed_z - z;
    
    if (penetration > 0.0) {
        // 法線力
        doublereal Fz = K_seabed * penetration - C_seabed * velocity.dGet(3);
        F_seabed = Vec3(0.0, 0.0, std::max(0.0, Fz));
    }
    
    return F_seabed;
}

// ========= 仮想ノード更新 =========
void ModuleCatenaryLM::UpdateVirtualNodes(doublereal dt)
{
    if (virtual_nodes.empty()) return;

    const Vec3& fairlead_pos = g_pNode->GetXCurr();
    const Vec3& fairlead_vel = g_pNode->GetVCurr();
    Vec3 anchor_pos(APx, APy, APz);
    Vec3 anchor_vel(0.0, 0.0, 0.0); // アンカーは固定
    
    doublereal segment_length = L / static_cast<doublereal>(NUM_SEGMENTS);
    doublereal ramp_factor = GetRampFactor(simulation_time);
    
    // 各仮想ノードの力計算と更新
    for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
        if (!virtual_nodes[i].active) continue;
        
        Vec3 F_total(0.0, 0.0, 0.0);
        
        // 重力
        Vec3 F_gravity(0.0, 0.0, -virtual_nodes[i].mass * g_gravity * ramp_factor);
        F_total += F_gravity;
        
        // 前のセグメントからの軸力
        if (i == 0) {
            // フェアリーダーとの接続
            Vec3 F_axial_prev = ComputeAxialForce(fairlead_pos, virtual_nodes[i].position, 
                                                fairlead_vel, virtual_nodes[i].velocity, segment_length);
            F_total += F_axial_prev * ramp_factor;
        } else {
            // 前の仮想ノードとの接続
            Vec3 F_axial_prev = ComputeAxialForce(virtual_nodes[i-1].position, virtual_nodes[i].position,
                                                virtual_nodes[i-1].velocity, virtual_nodes[i].velocity, segment_length);
            F_total += F_axial_prev * ramp_factor;
        }
        
        // 次のセグメントからの軸力
        if (i == virtual_nodes.size() - 1) {
            // アンカーとの接続
            Vec3 F_axial_next = ComputeAxialForce(virtual_nodes[i].position, anchor_pos,
                                                virtual_nodes[i].velocity, anchor_vel, segment_length);
            F_total -= F_axial_next * ramp_factor;
        } else {
            // 次の仮想ノードとの接続
            Vec3 F_axial_next = ComputeAxialForce(virtual_nodes[i].position, virtual_nodes[i+1].position,
                                                virtual_nodes[i].velocity, virtual_nodes[i+1].velocity, segment_length);
            F_total -= F_axial_next * ramp_factor;
        }
        
        // 海底反力
        Vec3 F_seabed = ComputeSeabedForce(virtual_nodes[i].position, virtual_nodes[i].velocity);
        F_total += F_seabed * ramp_factor;
        
        // 簡易減衰
        Vec3 F_damping = virtual_nodes[i].velocity * (-virtual_nodes[i].mass * 0.1);
        F_total += F_damping;
        
        // 加速度計算
        if (virtual_nodes[i].mass > 1e-12) {
            virtual_nodes[i].acceleration = F_total / virtual_nodes[i].mass;
            
            // 加速度制限
            doublereal acc_norm = virtual_nodes[i].acceleration.Norm();
            doublereal max_acc = 10.0 * g_gravity;
            if (acc_norm > max_acc) {
                virtual_nodes[i].acceleration = virtual_nodes[i].acceleration * (max_acc / acc_norm);
            }
        }
        
        // 速度・位置更新（Forward Euler）
        virtual_nodes[i].velocity += virtual_nodes[i].acceleration * dt;
        
        // 速度制限
        doublereal vel_norm = virtual_nodes[i].velocity.Norm();
        doublereal max_vel = 20.0;
        if (vel_norm > max_vel) {
            virtual_nodes[i].velocity = virtual_nodes[i].velocity * (max_vel / vel_norm);
        }
        
        virtual_nodes[i].position += virtual_nodes[i].velocity * dt;
        
        // 海底貫入防止
        if (virtual_nodes[i].position.dGet(3) < seabed_z) {
            virtual_nodes[i].position = Vec3(virtual_nodes[i].position.dGet(1),
                                           virtual_nodes[i].position.dGet(2),
                                           seabed_z + 0.01);
            if (virtual_nodes[i].velocity.dGet(3) < 0.0) {
                virtual_nodes[i].velocity = Vec3(virtual_nodes[i].velocity.dGet(1),
                                               virtual_nodes[i].velocity.dGet(2),
                                               0.0);
            }
        }
    }
}

// ========= ランプドマス法による総合力計算 =========
void ModuleCatenaryLM::ComputeLumpedMassForces(Vec3& F_total, Vec3& M_total)
{
    F_total = Vec3(0.0, 0.0, 0.0);
    M_total = Vec3(0.0, 0.0, 0.0);
    
    if (virtual_nodes.empty() || !virtual_nodes[0].active) {
        return;
    }
    
    const Vec3& fairlead_pos = g_pNode->GetXCurr();
    const Vec3& fairlead_vel = g_pNode->GetVCurr();
    
    doublereal segment_length = L / static_cast<doublereal>(NUM_SEGMENTS);
    doublereal ramp_factor = GetRampFactor(simulation_time);
    
    // フェアリーダーから最初の仮想ノードへの軸力
    Vec3 F_axial = ComputeAxialForce(fairlead_pos, virtual_nodes[0].position,
                                   fairlead_vel, virtual_nodes[0].velocity, segment_length);
    
    F_total = F_axial * ramp_factor;
    M_total = Vec3(0.0, 0.0, 0.0); // モーメントは考慮しない
}

// ========= MBDyn インターフェース =========
void ModuleCatenaryLM::Output(OutputHandler& OH) const
{
    if (bToBeOutput()) {
        if (OH.UseText(OutputHandler::LOADABLE)) {
            const Vec3& FP = g_pNode->GetXCurr();
            
            OH.Loadable() << GetLabel()
                << " " << FP.dGet(1)      // フェアリーダー位置
                << " " << FP.dGet(2)
                << " " << FP.dGet(3)
                << " " << virtual_nodes.size()  // 仮想ノード数
                << " " << simulation_time       // シミュレーション時間
                << " " << GetRampFactor(simulation_time)  // ランプファクター
                << std::endl;
        }
    }
}

void ModuleCatenaryLM::WorkSpaceDim(integer* piNumRows, integer* piNumCols) const 
{
    *piNumRows = 6;
    *piNumCols = 6;
}

VariableSubMatrixHandler& ModuleCatenaryLM::AssJac(
    VariableSubMatrixHandler& WorkMat,
    doublereal dCoef, 
    const VectorHandler& XCurr,
    const VectorHandler& XPrimeCurr)
{
    // ランプドマス法では簡略化されたヤコビアン
    WorkMat.SetNullMatrix();
    return WorkMat;
}

SubVectorHandler& ModuleCatenaryLM::AssRes(
    SubVectorHandler& WorkVec,
    doublereal dCoef,
    const VectorHandler& XCurr, 
    const VectorHandler& XPrimeCurr)
{
    integer iNumRows = 0;
    integer iNumCols = 0;
    WorkSpaceDim(&iNumRows, &iNumCols);
    WorkVec.ResizeReset(iNumRows);

    // インデックス設定
    integer iFirstMomIndex = g_pNode->iGetFirstMomentumIndex();
    for (int iCnt = 1; iCnt <= 6; iCnt++) {
        WorkVec.PutRowIndex(iCnt, iFirstMomIndex + iCnt);
    }

    // 現在時刻更新
    doublereal current_time = dCoef; // 簡略化：dCoefを時間として使用
    if (current_time > prev_time) {
        simulation_time = current_time;
        doublereal dt = current_time - prev_time;
        
        // 仮想ノード更新
        if (dt > 1e-12 && dt < 0.1) { // 妥当なタイムステップのみ
            UpdateVirtualNodes(dt);
        }
        
        prev_time = current_time;
    }

    // Force Scale Factor適用
    doublereal dFSF = FSF.dGet();
    
    // ランプドマス法による力計算
    Vec3 F, M;
    ComputeLumpedMassForces(F, M);
    
    F *= dFSF;
    M *= dFSF;

    // 力とモーメントをMBDynソルバーへ引き渡し
    WorkVec.Add(1, F);
    WorkVec.Add(4, M);

    return WorkVec;
}

unsigned int ModuleCatenaryLM::iGetNumPrivData(void) const
{
    return 0;
}

int ModuleCatenaryLM::iGetNumConnectedNodes(void) const
{
    return 1;
}

void ModuleCatenaryLM::GetConnectedNodes(std::vector<const Node *>& connectedNodes) const
{
    connectedNodes.resize(1);
    connectedNodes[0] = g_pNode;
}

void ModuleCatenaryLM::SetValue(DataManager *pDM, VectorHandler& X, VectorHandler& XP, SimulationEntity::Hints *ph)
{
    NO_OP;
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
    // should not be called, since initial workspace is empty
    ASSERT(0);
    WorkMat.SetNullMatrix();
    return WorkMat;
}

SubVectorHandler& ModuleCatenaryLM::InitialAssRes(
    SubVectorHandler& WorkVec, 
    const VectorHandler& XCurr)
{
    // should not be called, since initial workspace is empty
    ASSERT(0);
    WorkVec.ResizeReset(0);
    return WorkVec;
}

// ========= モジュール登録 =========
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
