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
    doublereal L_orig;
    doublereal w_orig;
    doublereal xacc;
    doublereal APx_orig, APy_orig, APz_orig;
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
        Cseabed(1.0e1),
        RampGravity(10.0),
        simulation_time(0.0),
        prev_time(0.0)
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

// ====== 重力ランプ係数計算 =======
doublereal ModuleCatenaryLM::GetGravityRampFactor(doublereal current_time) const {
    if (current_time >= RampGravity) {
        return 1.0;
    } else if (current_time <= 0.0) {
        return 0.0;
    } else {
        return 0.5*(1.0 - std::cos(3.14159 * current_time / RampGravity));
    }
}

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
    for (unsigned int j = 0; j < virtual_nodes.size(); ++j) {
        const VirtualNode& node = virtual_nodes[j];
        if (node.active) active_nodes++;
    }
    
    std::cout << "Virtual nodes initialized: " << active_nodes << "/" << virtual_nodes.size() << std::endl;
}

// ========= 仮想ノード更新 =========
void ModuleCatenaryLM::UpdateVirtualNodes(doublereal dCoef) {
    if (virtual_nodes.empty()) return;

    static bool first_call = true;
    static doublereal last_update_time = 0.0;
    
    if (first_call) {
        simulation_time = 0.0;
        last_update_time = 0.0;
        first_call = false;
    } else {
        doublereal dt_estimate = 0.001;
        if (dCoef > 1e-12) {
            dt_estimate = std::min(0.005, std::max(1e-6, dCoef));
        }
        simulation_time += dt_estimate;
    }

    doublereal gravity_factor = GetGravityRampFactor(simulation_time);

    // 適応タイムステップ
    doublereal dt = ComputeAdaptiveTimeStep();
    dt = std::min(dt, 0.005);
    dt = std::max(dt, 1e-6);

    // フェアリーダーとアンカー位置
    const Vec3& fairlead_pos = fairlead_node->GetXCurr();
    Vec3 anchor_pos(APx_orig, APy_orig, APz_orig);
    doublereal segment_length = L_orig / static_cast<doublereal>(Seg_param);
    
    // 各内部ノードに作用する力の計算
    for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
        VirtualNode& node = virtual_nodes[i];
        if (!node.active) continue;

        Vec3 F_total(0.0, 0.0, 0.0);
        
        // 重力
        Vec3 F_gravity(0.0, 0.0, - node.mass * g_gravity_param * gravity_factor);
        F_total += F_gravity;

        // 線形減衰
        doublereal damping_coeff = MooringCA / static_cast<doublereal>(Seg_param);
        Vec3 F_damping = node.velocity * (- damping_coeff);
        F_total += F_damping;

        // 弾性復元力：前のノード番号からの影響
        Vec3 prev_pos = (i == 0) ? fairlead_pos : virtual_nodes[i-1].position;
        Vec3 dx_prev = node.position - prev_pos;
        doublereal l_prev = dx_prev.Norm();
        
        if (l_prev > 1e-12) {
            Vec3 unit_prev = dx_prev / l_prev;
            doublereal strain_prev = (l_prev - segment_length) / segment_length;
            
            doublereal spring_force = MooringEA / static_cast<doublereal>(Seg_param);
            if (std::fabs(strain_prev) > 0.1) {
                spring_force *= (1.0 + std::fabs(strain_prev) * 0.5);
            }
            
            doublereal F_spring_prev = spring_force * strain_prev;
            F_total -= unit_prev * F_spring_prev;
        }

        // 弾性復元力：後ろのノード番号からの影響
        Vec3 next_pos = (i == virtual_nodes.size() - 1) ? anchor_pos : virtual_nodes[i+1].position;
        Vec3 dx_next = next_pos - node.position;
        doublereal l_next = dx_next.Norm();
        
        if (l_next > 1e-12) {
            Vec3 unit_next = dx_next / l_next;
            doublereal strain_next = (l_next - segment_length) / segment_length;
            
            doublereal spring_force = MooringEA / static_cast<doublereal>(Seg_param);
            if (std::fabs(strain_next) > 0.1) {
                spring_force *= (1.0 + std::fabs(strain_next) * 0.5);
            }
            
            doublereal F_spring_next = spring_force * strain_next;
            F_total += unit_next * F_spring_next;
        }

        // 海底相互作用
        if (node.position.dGet(3) <= seabed_z_param) {
            doublereal penetration = seabed_z_param - node.position.dGet(3);
            
            // 垂直反力
            doublereal normal_force = Kseabed * penetration * std::sqrt(std::max(penetration, 0.0));
            F_total += Vec3(0.0, 0.0, normal_force);
            
            // 摩擦力
            Vec3 vel_horizontal(node.velocity.dGet(1), node.velocity.dGet(2), 0.0);
            doublereal vel_h_norm = vel_horizontal.Norm();
            
            if (vel_h_norm > 1e-6) {
                doublereal mu = 0.3;  // 摩擦係数
                doublereal friction_mag = std::min(mu * normal_force, Cseabed * vel_h_norm);
                F_total -= vel_horizontal * (friction_mag / vel_h_norm);
            }
            
            // 垂直減衰
            if (virtual_nodes[i].velocity.dGet(3) < 0.0) {
                F_total += Vec3(0.0, 0.0, - node.velocity.dGet(3) * Cseabed);
            }
        }

        // TODO 流体力
        // waves ファイルから当てる？ Morison 方程式でも当てれる可能性

        // 数値積分（Velocity-Verlet 法）TODO 他に良い方法があれば検討：ここで速度制限を厳しくかけても良いかも
        if (node.mass > 1e-12) {
            Vec3 acceleration = F_total / node.mass;
            
            // 加速度制限
            doublereal acc_norm = acceleration.Norm();
            doublereal max_acc = 20.0 * g_gravity_param;
            if (acc_norm > max_acc) {
                acceleration *= (max_acc / acc_norm);
            }
            
            // Velocity-Verlet 更新
            node.position += node.velocity * dt + node.acceleration * (0.5 * dt * dt);
            
            Vec3 new_acceleration = acceleration;
            node.velocity += (node.acceleration + new_acceleration) * (0.5 * dt);
            node.acceleration = new_acceleration;
            
            // 速度制限
            doublereal vel_norm = node.velocity.Norm();
            doublereal max_vel = 50.0;
            if (vel_norm > max_vel) {
                node.velocity *= (max_vel / vel_norm);
            }
        }

        // 境界条件
        if (node.position.dGet(3) < seabed_z_param) {
            node.position = Vec3(
                node.position.dGet(1),
                node.position.dGet(2),
                seabed_z_param
            );
            
            if (node.velocity.dGet(3) < 0.0) {
                node.velocity = Vec3(
                    node.velocity.dGet(1) * 0.7,
                    node.velocity.dGet(2) * 0.7,
                    0.0
                );
            }
        }

    }
}

// ====== カテナリー理論に基づいて FP に作用する力の計算：module-catenary では AssRes の中で計算 ======
void ModuleCatenaryLM::ComputeCatenaryForces(Vec3& F_mooring, Vec3& M_mooring) const {
    const Vec3& FP = fairlead_node->GetXCurr();
    Vec3 AP(APx_orig, APy_orig, APz_orig);
    
    Vec3 FP_AP = FP - AP;
    doublereal h = std::fabs(FP_AP.dGet(3));
    doublereal L_APFP = std::sqrt(std::pow(FP_AP.dGet(1), 2.0) + std::pow(FP_AP.dGet(2), 2.0));
    
    if (h < 1e-6) {
        F_mooring = Vec3(0.0, 0.0, 0.0);
        M_mooring = Vec3(0.0, 0.0, 0.0);
        return;
    }
    
    doublereal L0_APFP = L_orig - h;
    doublereal delta = L_APFP - L0_APFP;
    doublereal d = delta / h;
    doublereal l = L_orig / h;
    
    doublereal H = 0.0, V = 0.0;
    doublereal p0 = 0.0;
    
    try {
        if (d <= 0.0) {
            H = 0.0;
            V = w_orig * h;
        } else if (d >= (std::sqrt(std::max(std::pow(l, 2.0) - 1.0, 0.0)) - (l - 1.0))) {
            // 係留索が張りきった状態
            H = MooringEA * delta / L_orig;
            V = w_orig * L_orig;
        } else {
            // カテナリー方程式を解く
            doublereal x1 = 1e-6;
            doublereal x2 = 1e6;
            doublereal Ans_x = rtsafe_catenary_local(x1, x2, xacc, d, l, p0);
            
            doublereal S = h * std::sqrt(1.0 + 2.0 * Ans_x);
            H = Ans_x * w_orig * h;
            V = w_orig * S;
        }
    } catch (...) {
        // フォールバック計算
        H = MooringEA * std::max(0.0, delta) / L_orig * 0.1;
        V = w_orig * L_orig * 0.5;
    }

    // Force Scale Factor適用
    doublereal fsf = FSF_orig.dGet();
    H *= fsf;
    V *= fsf;

    // 方向計算
    doublereal dFx = 0.0, dFy = 0.0;
    if (L_APFP > 1e-12) {
        doublereal angle = std::atan2(FP_AP.dGet(2), FP_AP.dGet(1));
        dFx = H * std::cos(angle);
        dFy = H * std::sin(angle);
    }
    
    // 係留力（フェアリーダーにかかる反力）
    F_mooring = Vec3(-dFx, -dFy, -V);
    M_mooring = Vec3(0.0, 0.0, 0.0);
}

// ========= 適応タイムステップ計算 =========
doublereal ModuleCatenaryLM::ComputeAdaptiveTimeStep() const {
    doublereal max_vel = 1e-6;
    doublereal max_acc = 1e-6;
    
    for (const auto& node : virtual_nodes) {
        if (!node.active) continue;
        max_vel = std::max(max_vel, node.velocity.Norm());
        max_acc = std::max(max_acc, node.acceleration.Norm());
    }
    
    doublereal segment_length = L_orig / static_cast<doublereal>(Seg_param);
    doublereal sound_speed = std::sqrt(MooringEA / (w_orig / g_gravity_param));
    
    // CFL条件
    doublereal dt_cfl = 0.3 * segment_length / sound_speed;
    
    // 速度制限
    doublereal dt_vel = 0.1 * segment_length / max_vel;
    
    // 加速度制限
    doublereal dt_acc = std::sqrt(2.0 * segment_length / max_acc);
    
    doublereal dt_min = dt_cfl;
    if (dt_vel < dt_min) dt_min = dt_vel;
    if (dt_acc < dt_min) dt_min = dt_acc;
    if (0.01 < dt_min) dt_min = 0.01;
    
    return dt_min;
}

// ========= MBDyn インターフェース =========
void ModuleCatenaryLM::WorkSpaceDim(integer* piNumRows, integer* piNumCols) const {
    *piNumRows = 6;
    *piNumCols = 6;
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

// ========= 残差ベクトル : AssRes =========
SubVectorHandler& ModuleCatenaryLM::AssRes(
    SubVectorHandler& WorkVec,
    doublereal dCoef,
    const VectorHandler& XCurr,
    const VectorHandler& XPrimeCurr
) {
    if (fairlead_node == nullptr) {
        WorkVec.ResizeReset(0);
        return WorkVec;
    }

    WorkVec.ResizeReset(6);

    integer iFirstMomIndex = fairlead_node->iGetFirstMomentumIndex();
    for (int iCnt = 1; iCnt <= 6; iCnt++) {
        WorkVec.PutRowIndex(iCnt, iFirstMomIndex + iCnt);
    }

    // 仮想ノード更新
    try {
        UpdateVirtualNodes(dCoef);
    } catch (...) {
        // 更新失敗時も継続
    }

    // カテナリー力計算
    Vec3 F_catenary, M_catenary;
    ComputeCatenaryForces(F_catenary, M_catenary);
    
    // 仮想ノードからの追加力
    Vec3 F_virtual(0.0, 0.0, 0.0);
    if (!virtual_nodes.empty() && virtual_nodes[0].active) {
        const Vec3& fairlead_pos = fairlead_node->GetXCurr();
        Vec3 dx = virtual_nodes[0].position - fairlead_pos;
        doublereal l = dx.Norm();
        
        if (l > 1e-12) {
            Vec3 t = dx / l;
            doublereal segment_length = L_orig / static_cast<doublereal>(Seg_param);
            doublereal strain = (l - segment_length) / segment_length;
            
            if (strain > -0.5) {  // 圧縮制限
                doublereal F_el = MooringEA * strain / static_cast<doublereal>(Seg_param);
                F_virtual = t * F_el;
            }
        }
    }
    
    Vec3 F_total = F_catenary + F_virtual;
    Vec3 M_total = M_catenary;

    WorkVec.Add(1, F_total);
    WorkVec.Add(4, M_total);

    return WorkVec;
}

// ========= 全体ヤコビアン：AssJac =========
VariableSubMatrixHandler& ModuleCatenaryLM::AssJac(
    VariableSubMatrixHandler& WorkMat,
    doublereal dCoef,
    const VectorHandler& XCurr,
    const VectorHandler& XPrimeCurr
) {
    if (fairlead_node == nullptr) {
        WorkMat.SetNullMatrix();
        return WorkMat;
    }

    FullSubMatrixHandler& Jac = WorkMat.SetFull();
    Jac.ResizeReset(6, 6);

    integer iFirstMomIndex = fairlead_node->iGetFirstMomentumIndex();
    integer iFirstPosIndex = fairlead_node->iGetFirstPositionIndex();
    
    for (int i = 0; i < 6; ++i) {
        Jac.PutRowIndex(i + 1, iFirstMomIndex + i + 1);
        Jac.PutColIndex(i + 1, iFirstPosIndex + i + 1);
    }

    // 係留系剛性計算
    const Vec3& FP = fairlead_node->GetXCurr();
    Vec3 AP(APx_orig, APy_orig, APz_orig);
    Vec3 FP_AP = FP - AP;
    doublereal L_APFP = FP_AP.Norm();
    
    if (L_APFP > 1e-12) {
        // 接線剛性
        doublereal k_tangent = MooringEA / L_orig;
        Vec3 n = FP_AP / L_APFP;
        
        // 位置剛性マトリックス
        for (int i = 1; i <= 3; ++i) {
            for (int j = 1; j <= 3; ++j) {
                doublereal k_ij = k_tangent * n.dGet(i) * n.dGet(j);
                
                // 幾何剛性項
                if (i == j) {
                    k_ij += k_tangent * 0.1 * (1.0 - n.dGet(i) * n.dGet(i));
                }
                
                Jac.PutCoef(i, j, k_ij);
            }
        }
    } else {
        // 対角剛性
        doublereal k_diag = MooringEA / L_orig * 0.1;
        for (int i = 1; i <= 3; ++i) {
            Jac.PutCoef(i, i, k_diag);
        }
    }

    // 回転自由度（小さな値で数値安定性確保）
    for (int i = 4; i <= 6; ++i) {
        Jac.PutCoef(i, i, 1e-12);
    }

    return WorkMat;
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
        OH.Loadable() << lbl << " [Error] Node not available" << std::endl;
        return;
    }

    // フェアリード位置
    const Vec3& fairlead_pos = fairlead_node->GetXCurr();
    
    // カテナリー力計算
    Vec3 F_mooring, M_mooring;
    ComputeCatenaryForces(F_mooring, M_mooring);
    
    // 仮想ノードからの追加力
    Vec3 F_virtual(0.0, 0.0, 0.0);
    if (!virtual_nodes.empty() && virtual_nodes[0].active) {
        Vec3 dx = virtual_nodes[0].position - fairlead_pos;
        doublereal l = dx.Norm();
        
        if (l > 1e-12) {
            Vec3 t = dx / l;
            doublereal segment_length = L_orig / static_cast<doublereal>(Seg_param);
            doublereal strain = (l - segment_length) / segment_length;
            
            if (strain > -0.5) {
                doublereal F_el = MooringEA * strain / static_cast<doublereal>(Seg_param);
                F_virtual = t * F_el;
            }
        }
    }
    
    Vec3 F_total = F_mooring + F_virtual;

    // 出力: ラベル FP位置(3) 外力(3)
    OH.Loadable() << lbl << " "
                  << fairlead_pos.dGet(1) << " "
                  << fairlead_pos.dGet(2) << " "
                  << fairlead_pos.dGet(3) << " "
                  << F_total.dGet(1) << " "
                  << F_total.dGet(2) << " "
                  << F_total.dGet(3) << std::endl;
}

// ========= モジュール登録 =========
bool catenary_lm_set(void) {
#ifdef DEBUG
    std::cerr << __FILE__ << ":" << __LINE__ << ": bool catenary_lm_set(void)" << std::endl;
#endif
    UserDefinedElemRead *rf = new UDERead<ModuleCatenaryLM>; // 新しいクラス名でテンプレート特殊化
    if (!SetUDE("catenary_lm", rf)) {
        delete rf;
        return false;
    }
    return true;
}

#ifndef STATIC_MODULES // MBDyn の標準的な動的モジュールロードの仕組み
extern "C" {
    int module_init(const char *module_name, void *pdm /*DataManager* */, void *php /* MBDynParser* */) {
        if (!catenary_lm_set()) { // 新しいセット関数を呼ぶ
            return -1; // 失敗
        }
        return 0; // 成功
    }
} // extern "C"
#endif // ! STATIC_MODULES
