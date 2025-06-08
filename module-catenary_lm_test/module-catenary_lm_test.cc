#include "mbconfig.h"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <iomanip>
#include <limits>

#include "dataman.h"
#include "userelem.h"
#include "module-catenary_lm.h"

#include "elem.h"
#include "strnode.h"
#include "drive.h"
#include "node.h"
#include "gravity.h"

#include <vector>

// ========= 仮想ノード構造 =========
struct VirtualNode {
    Vec3 position;
    Vec3 velocity;
    Vec3 acceleration;
    doublereal mass;
    bool active;
    doublereal strain_energy;

    VirtualNode() :
        position(0.0, 0.0, 0.0),
        velocity(0.0, 0.0, 0.0),
        acceleration(0.0, 0.0, 0.0),
        mass(0.0),
        active(true),
        strain_energy(0.0)
    {}

    VirtualNode(const Vec3& pos, doublereal m) :
        position(pos),
        velocity(0.0, 0.0, 0.0),
        acceleration(0.0, 0.0, 0.0),
        mass(m),
        active(true),
        strain_energy(0.0)
    {}
};

// ========= クラス定義 =========
class ModuleCatenaryLM : virtual public Elem, public UserDefinedElem
{
public: 
    ModuleCatenaryLM(unsigned uLabel, const DofOwner *pD0, DataManager* pDM, MBDynParser& HP);
    virtual ~ModuleCatenaryLM(void);

    virtual void Output(OutputHandler& OH) const;
    virtual void WorkSpaceDim(integer* piNumRows, integer* piNumCols) const;

    VariableSubMatrixHandler&
    AssJac(VariableSubMatrixHandler& WorkMat, 
        doublereal dCoef, 
        const VectorHandler& XCurr, 
        const VectorHandler& XPrimeCurr
    );

    SubVectorHandler&
    AssRes(SubVectorHandler& WorkVec, 
        doublereal dCoef, 
        const VectorHandler& XCurr, 
        const VectorHandler& XPrimeCurr
    );

    virtual void SetValue(DataManager* pDM,
        VectorHandler& X,
        VectorHandler& XP,
        SimulationEntity::Hints* pHints
    ) override;

    virtual unsigned int iGetNumConnectedNodes(void) const;
    virtual void SetInitialValue(VectorHandler& X, VectorHandler& XP);
    virtual std::ostream& Restart(std::ostream& out) const;

    virtual unsigned int iGetInitialNumDof(void) const;
    virtual void InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const;
    virtual VariableSubMatrixHandler&
    InitialAssJac(VariableSubMatrixHandler& WorkMat, const VectorHandler& XCurr);
    virtual SubVectorHandler&
    InitialAssRes(SubVectorHandler& WorkVec, const VectorHandler& XCurr);

    virtual const Node* pGetNode(unsigned int i) const;
    virtual Node* pGetNode(unsigned int i);

private:
    doublereal APx_orig, APy_orig, APz_orig;
    doublereal L_orig;
    doublereal w_orig;
    doublereal xacc;
    doublereal g_gravity_param;

    doublereal MooringEA;
    doublereal MooringCA;
    doublereal seabed_z_param;
    doublereal Kseabed;
    doublereal Cseabed;

    doublereal RampGravity;
    doublereal simulation_time;
    doublereal prev_time;

    DriveOwner FSF_orig;
    StructDispNode* fairlead_node;
    std::vector<VirtualNode> virtual_nodes;

    // セグメント数
    static const unsigned int Seg_param = 20;

    static double myasinh_local(double val);
    static double myacosh_local(double val);
    static void funcd_catenary_local(double x_param, double xacc, double &f_val, double &df_val, double d_geom, double l_geom, double &p0_angle);
    static double rtsafe_catenary_local(double x1_bounds, double x2_bounds, double xacc, double d_geom, double l_geom, double &p0_angle);

    void InitializeVirtualNodesFromCatenary();
    void UpdateVirtualNodes(doublereal dCoef);
    doublereal GetGravityRampFactor(doublereal current_time) const;
    void ComputeCatenaryForces(Vec3& F_mooring, Vec3& M_mooring) const;
    doublereal ComputeAdaptiveTimeStep() const;
};

// ========= コンストラクタ（.usr ファイル形式に対応） =========
ModuleCatenaryLM::ModuleCatenaryLM(
    unsigned uLabel,
    const DofOwner *pD0,
    DataManager* pDM,
    MBDynParser& HP
)
    : Elem(uLabel, flag(0)), UserDefinedElem(uLabel, pD0),
        fairlead_node(nullptr),    
        L_orig(0.0),
        w_orig(0.0),
        xacc(1e-4),
        APx_orig(0.0), APy_orig(0.0), APz_orig(0.0),
        MooringEA(1.0e3),
        MooringCA(1.0e2),
        g_gravity_param(9.80665),
        seabed_z_param(-320.0),
        Kseabed(1.0e3),
        Cseabed(1.0e1)
{
    if (HP.IsKeyWord("help")) {
        silent_cout(
            "\n"
            "Module:     ModuleCatenaryLM\n"
            "Usage:      catenary_lm, \n"
            "                fairlead_node_label,\n"
            "                LineLength, total_length,\n"
            "                LineWeight, unit_weight,\n"
            "                Xacc, rtsafe_accuracy,\n"
            "                APx, APy, APz,\n"
            "              [ EA, axial_stiffness, ]\n"
            "              [ CA, axial_damping, ]\n"
            "              [ gravity, g, ]\n"
            "              [ seabed, z, base_z, k, stiffness, c, damping, ]\n"
            "              [ force scale factor, (DriveCaller), ]\n"
            "              [ output, (FLAG) ] ;\n"
            "\n"
            << std::endl
        );
        if (!HP.IsArg()) {
            throw NoErr(MBDYN_EXCEPT_ARGS);
        }
    }

    // .usr ファイル形式のパラメータ読み込み
    unsigned int fairlead_node_label = HP.GetInt();
    L_orig = HP.GetReal();
    w_orig = HP.GetReal();
    xacc = HP.GetReal();
    APx_orig = HP.GetReal();
    APy_orig = HP.GetReal();
    APz_orig = HP.GetReal();
    MooringEA= HP.GetReal();
    MooringCA = HP.GetReal();
    g_gravity_param = HP.GetReal();
    seabed_z_param = HP.GetReal();
    Kseabed = HP.GetReal();
    Cseabed = HP.GetReal();

    // Force Scale Factor
    if (HP.IsKeyWord("force" "scale" "factor")) {
        FSF_orig.Set(HP.GetDriveCaller());
    } else {
        FSF_orig.Set(new OneDriveCaller);
    }

    SetOutputFlag(pDM->fReadOutput(HP, Elem::LOADABLE));

    // フェアリーダーノード取得
    Node* rawNode = pDM->pFindNode(Node::STRUCTURAL, fairlead_node_label);
    if (rawNode == nullptr) {
        throw ErrGeneric(MBDYN_EXCEPT_ARGS);
    }
    fairlead_node = dynamic_cast<StructDispNode*>(rawNode);
    if (fairlead_node == nullptr) {
        throw ErrGeneric(MBDYN_EXCEPT_ARGS);
    }

    // 仮想ノード初期化
    virtual_nodes.resize(Seg_param - 1);
    
    doublereal segment_length = L_orig / static_cast<doublereal>(Seg_param);
    doublereal node_mass = (w_orig / g_gravity_param) * segment_length;

    for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
        virtual_nodes[i].mass = node_mass;
        virtual_nodes[i].active = true;
    }

    // カテナリー理論による初期位置設定
    InitializeVirtualNodesFromCatenary();

    pDM->GetLogFile() << "ModuleCatenaryLM (" << GetLabel() << ") initialized :" << std::endl;
    pDM->GetLogFile() << "  Fairlead Node Label: " << fairlead_node_label << std::endl;
    pDM->GetLogFile() << "  Anchor: (" << APx_orig << ", " << APy_orig << ", " << APz_orig << ")" << std::endl;
    pDM->GetLogFile() << "  Length: " << L_orig << ", Weight: " << w_orig << std::endl;
    pDM->GetLogFile() << "  Segments: " << Seg_param << std::endl;
    pDM->GetLogFile() << "  EA: " << MooringEA << ", CA: " << MooringCA << std::endl;
    pDM->GetLogFile() << "  Virtual Nodes: " << virtual_nodes.size() << std::endl;
}

ModuleCatenaryLM::~ModuleCatenaryLM(void) {}

// ========= カテナリー理論関数 =========
double ModuleCatenaryLM::myasinh_local(double val) { 
    return std::log(val + std::sqrt(val * val + 1.)); 
}

double ModuleCatenaryLM::myacosh_local(double val) { 
    if (val < 1.0) val = 1.0; 
    return std::log(val + std::sqrt(val * val - 1.));
}

// ========= カテナリー関数 =========
void ModuleCatenaryLM::funcd_catenary_local(double x_param, double xacc, double &f_val, double &df_val, double d_geom, double l_geom, double &p0_angle) {
    int i, max;
    double f1, df1;
    max = 1000;

    // 係留索が完全にたるんでいて，水平張力 0
    if (x_param == 0.0) {
        f_val = -d_geom;
        df_val = 0.0;
        p0_angle = 0.0;
        return;
    }

    // 水平張力あり
    if (x_param > 0.0) {

        // 全長が垂直距離以下という物理的にあり得ない状況だが，特定の計算方法
        if (l_geom <= 1.0) {
            double x_1 = 1.0/x_param + 1.0;
            f_val = x_param * myacosh_local(x_1) - std::sqrt(1.0 + 2.0*x_param) + 1.0 - d_geom;
            df_val = myacosh_local(x_1) - 1.0/std::sqrt(1.0 + 2.0*x_param) - 1.0/(x_param*std::sqrt(std::pow(x_1, 2.0) - 1.0));
            p0_angle = 0.0;
        } else {

            // 海底に接する可能性のあるケース
            if (x_param > (l_geom*l_geom - 1.0) / 2.0) {
                p0_angle = 0.0;
                for (int i = 1; i < max; i++) {
                    double func1 = 1.0/x_param + 1.0/std::cos(p0_angle);

                    f1 = x_param*(std::sqrt(std::pow(func1, 2.0) - 1.0) - std::tan(p0_angle)) - l_geom;
                    df1 = x_param*(func1*std::tan(p0_angle)*(1.0/std::cos(p0_angle))/std::sqrt(std::pow(func1, 2.0) - 1.0) - std::pow(std::tan(p0_angle), 2.0) - 1.0);
                    p0_angle = p0_angle - f1/df1;
                    
                    // 収束判定
                    func1 = 1.0/x_param + 1.0/std::cos(p0_angle);
                    f1 = x_param*(std::sqrt(std::pow(func1, 2.0) - 1.0) - std::tan(p0_angle)) - l_geom;

                    if (std::fabs(f1) < xacc) {
                        break;
                    }
                }
                
                // 収束しなかった場合：エラー
                if (std::fabs(f1) > xacc) {
                    std::cout << "ERROR: p0_angle iteration did not converge, fabs(f1) = " << std::fabs(f1) << " > " << xacc << std::endl;
                    throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
                }

                double x_2 = l_geom/x_param + std::tan(p0_angle);
                double x_3 = std::tan(p0_angle);

                f_val = x_param*(myasinh_local(x_2) - myasinh_local(x_3)) - l_geom + 1.0 - d_geom;
                df_val = myasinh_local(x_2) - myasinh_local(x_3) - l_geom/(x_param*std::sqrt(std::pow(x_2, 2.0) + 1.0));
            } else {

                // 海底に接触しない場合：単純なカテナリー
                double x_5 = 1.0/x_param + 1.0;

                f_val = x_param*myacosh_local(x_5) - std::sqrt(1.0 + 2.0*x_param) + 1.0 - d_geom;
                df_val = myacosh_local(x_5) - 1.0/std::sqrt(1.0 + 2.0*x_param) - 1.0/(x_param*std::sqrt(std::pow(x_5, 2.0) - 1.0));
                p0_angle = 0.0;
            }
        }
    } else {
        std::cout << "ERROR: x_param < 0" << std::endl;
        throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
    }
}

double ModuleCatenaryLM::rtsafe_catenary_local(double x1_bounds, double x2_bounds, double xacc, double d_geom, double l_geom, double &p0_angle) {
    const int MAXIT = 1000;
    int j;
    double fh, fl, xh, xl, df_val;
    double dx, dxold, f_val, temp, rts;
    double p1, p2;

    // 境界での関数値を計算
    ModuleCatenaryLM::funcd_catenary_local(x1_bounds, xacc, fl, df_val, d_geom, l_geom, p1);
    ModuleCatenaryLM::funcd_catenary_local(x2_bounds, xacc, fh, df_val, d_geom, l_geom, p2);

    // fl fh が同符号であれば根が無いため，中断する
    if ((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0)) {
        std::cout << "ERROR: Root is not bracketed. fl = " << fl << ", fh = " << fh << std::endl;
        throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
    }

    // fl または fh が 0 であれば x1 x2 が既に根であるため，その値を返す
    if (fl == 0.0) {
        p0_angle = p1;
        return x1_bounds;
    }
    if (fh == 0.0) {
        p0_angle = p2;
        return x2_bounds;
    }

    // f(xl) < 0 かつ f(xh) > 0 を満たすように根を含む区間の下限 xl と根を含む区間の上限 xh を設定
    if (fl < 0.0) {
        xl = x1_bounds;
        xh = x2_bounds;
    } else {
        xh = x1_bounds;
        xl = x2_bounds;
    }

    // 根の初期推定値は中点とする
    rts = 0.5*(x1_bounds + x2_bounds);
    dxold = std::fabs(x2_bounds - x1_bounds);  // 修正: x2_bounds - x2_bounds -> x2_bounds - x1_bounds
    dx = dxold;

    // funcd により初期推定値 rts における関数の値 f_val と微分 df_val を計算する
    ModuleCatenaryLM::funcd_catenary_local(rts, xacc, f_val, df_val, d_geom, l_geom, p0_angle);

    for (j = 0; j < MAXIT; j++) {
        // ニュートン法のステップが区間の外に出るかどうか || ニュートン法のステップが二分法のステップよりも効果的かどうか
        if (((rts - xh)*df_val - f_val)*((rts - xl)*df_val - f_val) > 0.0 || (std::fabs(2.0*f_val)) > std::fabs(dxold*df_val)) {
            
            // 二分法を採用した場合
            dxold = dx;
            dx = 0.5*(xh - xl);
            rts = xl + dx;
            if (xl == rts) {
                return rts;
            }
        } else {

            // ニュートン法を採用した場合
            dxold = dx;
            dx = f_val / df_val;
            temp = rts;
            rts -= dx;
            if (temp == rts) {
                return rts;
            }
        }

        // 収束判定
        if (std::fabs(dx) < xacc) { 
            return rts; 
        }

        // 新しい推定値 rts で funcd を呼び出し，関数値 f を更新
        ModuleCatenaryLM::funcd_catenary_local(rts, xacc, f_val, df_val, d_geom, l_geom, p0_angle);

        // 区間の更新
        if (f_val < 0.0) {
            xl = rts;
        } else {
            xh = rts;
        }
    }

    std::cout << "ERROR: Maximum iterations exceeded in bisection method" << std::endl;
    throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
}

// ========= カテナリー理論による内部ノードの初期位置 =========
void ModuleCatenaryLM::InitializeVirtualNodesFromCatenary() {
    if (fairlead_node == nullptr || virtual_nodes.empty()) return;

    const Vec3& fairlead_pos = fairlead_node->GetXCurr();
    Vec3 anchor_pos(APx_orig, APy_orig, APz_orig);
    
    // フェアリーダーとアンカー間のベクトル
    Vec3 FP_AP = fairlead_pos - anchor_pos;
    doublereal h = FP_AP.dGet(3);  // 垂直距離：符号付き
    doublereal L_APFP = std::sqrt(std::pow(FP_AP.dGet(1), 2.0) + std::pow(FP_AP.dGet(2), 2.0));
    
    // 水平方向の単位ベクトル
    Vec3 horizontal_dir(1.0, 0.0, 0.0);
    if (L_APFP > 1e-12) {
        horizontal_dir = Vec3(FP_AP.dGet(1) / L_APFP, FP_AP.dGet(2) / L_APFP, 0.0);
    }

    bool catenary_success = false;

    // カテナリー理論による計算
    if (std::fabs(h) > 1e-6 && L_APFP > 1e-6 && L_orig > 1e-6) {
        try {

            // 無次元パラメータの計算
            doublereal d_geom = (L_APFP - (L_orig - std::fabs(h))) / std::fabs(h);
            doublereal l_geom = L_orig / std::fabs(h);
            
            // 妥当性チェック
            if (l_geom > 1.0 && std::fabs(d_geom) < 10.0) {
                doublereal p0_angle = 0.0;
                doublereal x_param;
                
                // 水平張力パラメータの求解
                doublereal x1_bounds = 0.001;  // 下限
                doublereal x2_bounds = 1000.0; // 上限
                
                // Newton-Raphson 法で水平張力パラメータを求める
                x_param = rtsafe_catenary_local(x1_bounds, x2_bounds, xacc, d_geom, l_geom, p0_angle);
                
                if (x_param > 1e-12) {

                    // 物理量の計算
                    doublereal H = x_param * w_orig * std::fabs(h); // 水平張力
                    doublereal a = H / w_orig;// カテナリーパラメータ
                    
                    // ======== 座標系の設定 ========
                    // アンカーを原点としてフェアリーダー方向を x 正方向とする局所座標系
                    Vec3 local_origin = anchor_pos;
                    
                    // 各仮想ノードの位置を計算
                    doublereal segment_length = L_orig / static_cast<doublereal>(Seg_param);
                    
                    for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
                        // アンカーからの弧長距離
                        doublereal s = segment_length * static_cast<doublereal>(i + 1);
                        
                        doublereal x_local, z_local;
                        
                        // 海底接触の有無による場合分け
                        if (p0_angle > 1e-6) {
                            // 海底接触ありの場合
                            doublereal s_contact = a * std::tan(p0_angle);  // 海底接触開始点までの弧長
                            
                            if (s <= s_contact) {
                                // 海底上の部分
                                x_local = s;
                                z_local = seabed_z_param - anchor_pos.dGet(3);
                            } else {
                                // カテナリー部分
                                doublereal s_cat = s - s_contact;  // カテナリー部分の弧長
                                doublereal beta = s_cat / a;
                                
                                // カテナリー方程式
                                x_local = s_contact + a * myasinh_local(std::sinh(beta));
                                z_local = (seabed_z_param - anchor_pos.dGet(3)) + a * (std::cosh(beta) - 1.0);
                            }
                        } else {
                            // 海底接触なしの場合：純粋なカテナリー
                            // アンカー点での接線角度を計算
                            doublereal theta_0 = -myasinh_local(L_APFP / a - 1.0/x_param);
                            
                            // 弧長パラメータ
                            doublereal beta = s / a + theta_0;
                            
                            // カテナリー方程式による座標計算
                            x_local = a * (std::sinh(beta) - std::sinh(theta_0));
                            z_local = a * (std::cosh(beta) - std::cosh(theta_0));
                        }
                        
                        // 局所座標からグローバル座標への変換
                        Vec3 local_pos(x_local, 0.0, z_local);
                        virtual_nodes[i].position = local_origin + horizontal_dir * x_local + Vec3(0.0, 0.0, z_local);
                        virtual_nodes[i].velocity = Vec3(0.0, 0.0, 0.0);
                        virtual_nodes[i].active = true;
                        
                        // 座標の妥当性チェック
                        if (!std::isfinite(virtual_nodes[i].position.dGet(1)) ||
                            !std::isfinite(virtual_nodes[i].position.dGet(2)) ||
                            !std::isfinite(virtual_nodes[i].position.dGet(3))) {
                            throw ErrGeneric(MBDYN_EXCEPT_ARGS);
                        }
                    }
                    
                    catenary_success = true;
                    
                    std::cout << "Catenary initialization successful:" << std::endl;
                    std::cout << "  H = " << H << " N, a = " << a << " m" << std::endl;
                    std::cout << "  x_param = " << x_param << ", p0_angle = " << p0_angle << " rad" << std::endl;
                    std::cout << "  d_geom = " << d_geom << ", l_geom = " << l_geom << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Catenary calculation failed: " << e.what() << std::endl;
            catenary_success = false;
        } catch (...) {
            std::cerr << "Catenary calculation failed with unknown error" << std::endl;
            catenary_success = false;
        }
    }

    // フォールバック：線形補間（カテナリー計算が失敗した場合）
    if (!catenary_success) {
        std::cout << "Using linear interpolation fallback for virtual node initialization" << std::endl;
        
        for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
            doublereal ratio = static_cast<doublereal>(i + 1) / static_cast<doublereal>(Seg_param);
            virtual_nodes[i].position = anchor_pos + (fairlead_pos - anchor_pos) * ratio;
            
            // 軽微な垂れ下がりを追加
            doublereal sag = 0.05 * L_orig * ratio * (1.0 - ratio);
            virtual_nodes[i].position += Vec3(0.0, 0.0, -sag);
            
            virtual_nodes[i].velocity = Vec3(0.0, 0.0, 0.0);
            virtual_nodes[i].active = true;
        }
    }

    // 初期化完了の確認
    unsigned int active_nodes = 0;
    for (const auto& node : virtual_nodes) {
        if (node.active) active_nodes++;
    }
    
    std::cout << "Virtual nodes initialized: " << active_nodes << "/" << virtual_nodes.size() << std::endl;
}

// ========= 仮想ノード更新 =========
void ModuleCatenaryLM::UpdateVirtualNodes(doublereal dCoef) {
    if (virtual_nodes.empty()) return;

    const doublereal dt = 0.005;  // 固定小タイムステップ
    const doublereal damping_factor = 0.98;  // 強い数値減衰
    
    for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
        if (!virtual_nodes[i].active) continue;

        // 極小重力のみ（1/50に）
        Vec3 F_gravity(0.0, 0.0, -virtual_nodes[i].mass * g_gravity_param * 0.02);

        // 極小減衰
        Vec3 F_damping = virtual_nodes[i].velocity * (-virtual_nodes[i].mass * 0.5);

        // 極小海底反力（必要最小限）
        Vec3 F_seabed(0.0, 0.0, 0.0);
        if (virtual_nodes[i].position.dGet(3) < seabed_z_param) {
            doublereal pen = seabed_z_param - virtual_nodes[i].position.dGet(3);
            doublereal Fz = Kseabed * pen * 0.1;  // さらに1/10に
            F_seabed = Vec3(0.0, 0.0, std::max(0.0, Fz));
        }

        Vec3 F_total = F_gravity + F_damping + F_seabed;

        // 加速度計算と更新
        if (virtual_nodes[i].mass > 1e-12) {
            Vec3 acceleration = F_total / virtual_nodes[i].mass;
            
            // 強い制限
            doublereal acc_norm = acceleration.Norm();
            doublereal max_acc = 5.0 * g_gravity_param;
            if (acc_norm > max_acc) {
                acceleration = acceleration * (max_acc / acc_norm);
            }

            // 速度更新（強い減衰）
            virtual_nodes[i].velocity = virtual_nodes[i].velocity * damping_factor + acceleration * dt;
            
            // 速度制限
            doublereal vel_norm = virtual_nodes[i].velocity.Norm();
            doublereal max_vel = 5.0;
            if (vel_norm > max_vel) {
                virtual_nodes[i].velocity = virtual_nodes[i].velocity * (max_vel / vel_norm);
            }
            
            // 位置更新
            virtual_nodes[i].position += virtual_nodes[i].velocity * dt;
        }

        // 海底貫入防止
        if (virtual_nodes[i].position.dGet(3) < seabed_z_param) {
            virtual_nodes[i].position = Vec3(
                virtual_nodes[i].position.dGet(1), 
                virtual_nodes[i].position.dGet(2), 
                seabed_z_param + 0.01
            );
            if (virtual_nodes[i].velocity.dGet(3) < 0.0) {
                virtual_nodes[i].velocity = Vec3(
                    virtual_nodes[i].velocity.dGet(1),
                    virtual_nodes[i].velocity.dGet(2),
                    0.0
                );
            }
        }
    }
}

// ========= MBDyn インターフェース =========
void ModuleCatenaryLM::WorkSpaceDim(integer* piNumRows, integer* piNumCols) const {
    *piNumRows = 3;
    *piNumCols = 3;
}

unsigned int ModuleCatenaryLM::iGetNumConnectedNodes(void) const {
    return 1;
}

unsigned int ModuleCatenaryLM::iGetInitialNumDof(void) const {
    return 0;
}

const Node* ModuleCatenaryLM::pGetNode(unsigned int i) const {
    if (i == 1) return fairlead_node;
    return nullptr;
}

Node* ModuleCatenaryLM::pGetNode(unsigned int i) {
    if (i == 1) return fairlead_node;
    return nullptr;
}

void ModuleCatenaryLM::SetInitialValue(VectorHandler& X, VectorHandler& XP) {
    InitializeVirtualNodesFromCatenary();
}

// ========= 残差ベクトル =========
SubVectorHandler& ModuleCatenaryLM::AssRes(
    SubVectorHandler& R,
    doublereal dCoef,
    const VectorHandler& XCurr,
    const VectorHandler& XPrimeCurr
) {
    if (fairlead_node == nullptr) {
        R.ResizeReset(0);
        return R;
    }

    R.ResizeReset(3);

    integer iFirstIndex = fairlead_node->iGetFirstPositionIndex();
    R.PutRowIndex(1, iFirstIndex + 1);
    R.PutRowIndex(2, iFirstIndex + 2);
    R.PutRowIndex(3, iFirstIndex + 3);

    // 仮想ノード更新
    UpdateVirtualNodes(dCoef);

    // 極小係留力（重力の0.1%程度）
    const doublereal fsf = FSF_orig.dGet();
    Vec3 F_mooring(0.0, 0.0, -w_orig * L_orig * 0.001 * fsf);

    // 最初の仮想ノードとの極小軸力（オプション）
    if (!virtual_nodes.empty() && virtual_nodes[0].active) {
        const Vec3& fairlead_pos = fairlead_node->GetXCurr();
        Vec3 dx = virtual_nodes[0].position - fairlead_pos;
        doublereal l = dx.Norm();
        
        if (l > 1e-12) {
            Vec3 t = dx / l;
            doublereal segment_length = L_orig / static_cast<doublereal>(Seg_param);
            doublereal strain = (l - segment_length) / segment_length;
            
            if (strain > 0.0) {
                doublereal Fel = MooringEA * strain * 0.1;  // さらに1/10に
                F_mooring += t * (Fel * fsf);
            }
        }
    }

    R.PutCoef(1, F_mooring.dGet(1));
    R.PutCoef(2, F_mooring.dGet(2));
    R.PutCoef(3, F_mooring.dGet(3));

    return R;
}

// ========= ヤコビアン =========
VariableSubMatrixHandler& ModuleCatenaryLM::AssJac(
    VariableSubMatrixHandler& WH,
    doublereal dCoef,
    const VectorHandler& XCurr, 
    const VectorHandler& XPrimeCurr
) {
    if (fairlead_node == nullptr) {
        WH.SetNullMatrix();
        return WH;
    }

    FullSubMatrixHandler& K = WH.SetFull();
    K.ResizeReset(3, 3);

    integer iFirstIndex = fairlead_node->iGetFirstPositionIndex();
    for (int i = 0; i < 3; ++i) {
        K.PutRowIndex(i + 1, iFirstIndex + i + 1);
        K.PutColIndex(i + 1, iFirstIndex + i + 1);
    }

    // 極小剛性（対角のみ）
    doublereal k_stiff = MooringEA * 0.01;
    K.PutCoef(1, 1, k_stiff);
    K.PutCoef(2, 2, k_stiff);
    K.PutCoef(3, 3, k_stiff);

    return WH;
}

// ========= 初期条件関連 =========
void ModuleCatenaryLM::InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const {
    *piNumRows = 0;
    *piNumCols = 0;
}

SubVectorHandler& ModuleCatenaryLM::InitialAssRes(
    SubVectorHandler& WorkVec,
    const VectorHandler& XCurr
) {
    WorkVec.ResizeReset(0);
    return WorkVec;
}

VariableSubMatrixHandler& ModuleCatenaryLM::InitialAssJac(
    VariableSubMatrixHandler& WorkMat,
    const VectorHandler& XCurr
) {
    WorkMat.SetNullMatrix();
    return WorkMat;
}

// ========= 出力 =========
void ModuleCatenaryLM::Output(OutputHandler& OH) const {
    if (!bToBeOutput()) return;
    if (!OH.UseText(OutputHandler::LOADABLE)) return;

    const integer lbl = GetLabel();

    if (fairlead_node == nullptr) {
        OH.Loadable() << lbl << " [Error] Node not available\n";
        return;
    }

    const Vec3 posFL = fairlead_node->GetXCurr();

    OH.Loadable() << lbl << " FairleadPos "
                  << posFL.dGet(1) << " "
                  << posFL.dGet(2) << " "
                  << posFL.dGet(3) << " VirtualNodes " << virtual_nodes.size();

    // 最初の数個の仮想ノード位置を出力
    for (unsigned int i = 0; i < virtual_nodes.size() && i < 3; ++i) {
        if (virtual_nodes[i].active) {
            OH.Loadable() << " " << virtual_nodes[i].position.dGet(1)
                         << " " << virtual_nodes[i].position.dGet(2)
                         << " " << virtual_nodes[i].position.dGet(3);
        }
    }

    // 海底接触数
    unsigned int seabed_contact_count = 0;
    for (const auto& node : virtual_nodes) {
        if (node.position.dGet(3) <= seabed_z_param + 0.1) {
            seabed_contact_count++;
        }
    }
    OH.Loadable() << " SeabedContacts " << seabed_contact_count;

    OH.Loadable() << '\n';
}

std::ostream& ModuleCatenaryLM::Restart(std::ostream& out) const {
    out << "# ModuleCatenaryLM: Restart not implemented" << std::endl;
    return out;
}

void ModuleCatenaryLM::SetValue(
    DataManager* /*pDM*/,
    VectorHandler& /*X*/,
    VectorHandler& /*XP*/,
    SimulationEntity::Hints* /*pHints*/
) {
    // 空実装
}

// ========= モジュール登録 =========
bool catenary_lm_set(void) {
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
            return -1;
        }
        return 0;
    }
}
#endif
