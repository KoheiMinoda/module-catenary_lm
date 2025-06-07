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

// ========= 仮想ノード構造体（最小限版） =========
struct VirtualNodeMinimal {
    Vec3 position;
    Vec3 velocity;
    doublereal mass;
    bool active;

    VirtualNodeMinimal() :
        position(0.0, 0.0, 0.0),
        velocity(0.0, 0.0, 0.0),
        mass(0.0),
        active(true)
    {}

    VirtualNodeMinimal(const Vec3& pos, doublereal m) :
        position(pos),
        velocity(0.0, 0.0, 0.0),
        mass(m),
        active(true)
    {}
};

// ========= 最小限カテナリーランプドマス要素クラス =========
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
    // 基本パラメータ（.usr ファイルから読み込み）
    doublereal APx_orig, APy_orig, APz_orig;  // アンカー座標
    doublereal L_orig;                        // 全長
    doublereal w_orig;                        // 単位重量
    doublereal xacc_orig;                     // rtsafe精度
    doublereal g_gravity_param;               // 重力加速度

    // 極小化された物理パラメータ
    doublereal EA_minimal;                    // 極小軸剛性
    doublereal CA_minimal;                    // 極小減衰
    doublereal seabed_z_param;               // 海底Z座標
    doublereal Kseabed_minimal;              // 極小海底剛性
    doublereal Cseabed_minimal;              // 極小海底減衰

    // Force Scale Factor
    DriveOwner FSF_orig;

    // ノード
    StructDispNode* fairlead_node;
    std::vector<VirtualNodeMinimal> virtual_nodes;

    // セグメント数（固定・少なく）
    static const unsigned int Seg_param_minimal = 8;  // 少ないセグメント数

    // カテナリー理論関数（既存のものを簡略化）
    static double myasinh_local(double val);
    static double myacosh_local(double val);
    static void funcd_catenary_local(double x_param, double xacc, double &f_val, double &df_val, double d_geom, double l_geom, double &p0_angle);
    static double rtsafe_catenary_local(double x1_bounds, double x2_bounds, double xacc_tol, double d_geom, double l_geom, double &p0_angle_out);

    void InitializeVirtualNodesFromCatenary();
    void UpdateVirtualNodesMinimal(doublereal dCoef);
};

// ========= コンストラクタ（.usr ファイル形式に対応） =========
ModuleCatenaryLM::ModuleCatenaryLM(
    unsigned uLabel,
    const DofOwner *pD0,
    DataManager* pDM,
    MBDynParser& HP
)
    : Elem(uLabel, flag(0)), UserDefinedElem(uLabel, pD0),
      APx_orig(0.0), APy_orig(0.0), APz_orig(0.0),
      L_orig(0.0), w_orig(0.0), xacc_orig(1e-4),
      g_gravity_param(9.80665),
      EA_minimal(1.0e3),        // 元の値の1/1000000
      CA_minimal(1.0e2),        // 元の値の1/1000000
      seabed_z_param(-320.0),
      Kseabed_minimal(1.0e3),   // 元の値の1/10000
      Cseabed_minimal(1.0e1),   // 元の値の1/5000
      fairlead_node(nullptr)
{
    if (HP.IsKeyWord("help")) {
        silent_cout(
            "\n"
            "Module:     ModuleCatenaryLM (Minimal Lumped Mass Catenary)\n"
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

    // LineLength
    L_orig = HP.GetReal();

    // LineWeight
    w_orig = HP.GetReal();

    // Xacc
    xacc_orig = HP.GetReal();

    // アンカー座標（3つ連続で読み込み）
    APx_orig = HP.GetReal();
    APy_orig = HP.GetReal();
    APz_orig = HP.GetReal();

    // EA（軸剛性）- 極小値に設定
    EA_minimal = HP.GetReal();

    // CA（軸減衰）- 極小値に設定
    CA_minimal = HP.GetReal();

    // gravity
    g_gravity_param = HP.GetReal();
        
    seabed_z_param = HP.GetReal();
    Kseabed_minimal = HP.GetReal();
    Cseabed_minimal = HP.GetReal();

    // Force Scale Factor
    if (HP.IsKeyWord("force" "scale" "factor")) {
        FSF_orig.Set(HP.GetDriveCaller());
    } else {
        FSF_orig.Set(new OneDriveCaller);
    }

    // 出力フラグ
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

    // 仮想ノード初期化（少ないセグメント数）
    virtual_nodes.resize(Seg_param_minimal - 1);  // 7個の仮想ノード
    
    doublereal segment_length = L_orig / static_cast<doublereal>(Seg_param_minimal);
    doublereal node_mass = (w_orig / g_gravity_param) * segment_length * 0.05;  // 質量を1/20に

    for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
        virtual_nodes[i].mass = node_mass;
        virtual_nodes[i].active = true;
    }

    // カテナリー理論による初期位置設定
    InitializeVirtualNodesFromCatenary();

    pDM->GetLogFile() << "ModuleCatenaryLM (" << GetLabel() << ") initialized (MINIMAL VERSION):" << std::endl;
    pDM->GetLogFile() << "  Fairlead Node Label: " << fairlead_node_label << std::endl;
    pDM->GetLogFile() << "  Anchor: (" << APx_orig << ", " << APy_orig << ", " << APz_orig << ")" << std::endl;
    pDM->GetLogFile() << "  Length: " << L_orig << ", Weight: " << w_orig << std::endl;
    pDM->GetLogFile() << "  Segments: " << Seg_param_minimal << " (minimal)" << std::endl;
    pDM->GetLogFile() << "  EA_minimal: " << EA_minimal << ", CA_minimal: " << CA_minimal << std::endl;
    pDM->GetLogFile() << "  Virtual Nodes: " << virtual_nodes.size() << std::endl;
}

ModuleCatenaryLM::~ModuleCatenaryLM(void) {}

// ========= カテナリー理論関数（簡略版） =========
double ModuleCatenaryLM::myasinh_local(double val) { 
    return std::log(val + std::sqrt(val * val + 1.)); 
}

double ModuleCatenaryLM::myacosh_local(double val) { 
    if (val < 1.0) val = 1.0;  // 安全な値に修正
    return std::log(val + std::sqrt(val * val - 1.));
}

void ModuleCatenaryLM::funcd_catenary_local(double x_param, double xacc, double &f_val, double &df_val, double d_geom, double l_geom, double &p0_angle) {
    if(x_param <= 0.0) {
        f_val = -d_geom;
        df_val = 1.0;
        p0_angle = 0.0;
        return;
    }

    try {
        // 簡略化：単純なケースのみ処理
        double X_5_internal = 1.0/x_param + 1.0;
        if (X_5_internal < 1.0) X_5_internal = 1.0;
        
        double term_sqrt1 = 1.0 + 2.0*x_param;
        if (term_sqrt1 < 0.0) term_sqrt1 = 0.0;
        
        f_val = x_param*myacosh_local(X_5_internal) - std::sqrt(term_sqrt1) + 1.0 - d_geom;
        df_val = myacosh_local(X_5_internal) - 1.0/std::sqrt(std::max(term_sqrt1, 1e-12));
        p0_angle = 0.0;
    } catch (...) {
        f_val = -d_geom;
        df_val = 1.0;
        p0_angle = 0.0;
    }
}

double ModuleCatenaryLM::rtsafe_catenary_local(double x1_bounds, double x2_bounds, double xacc_tol, double d_geom, double l_geom, double &p0_angle_out) {
    const int MAXIT = 50;  // 反復回数を制限
    
    try {
        double fl, fh, xl, xh, dx, rts, f, df;
        double p1, p2;
        
        funcd_catenary_local(x1_bounds, xacc_tol, fl, df, d_geom, l_geom, p1);
        funcd_catenary_local(x2_bounds, xacc_tol, fh, df, d_geom, l_geom, p2);
        
        if((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0)) {
            p0_angle_out = 0.0;
            return 0.5 * (x1_bounds + x2_bounds);
        }
        
        if(fl == 0.0) { p0_angle_out = p1; return x1_bounds; }
        if(fh == 0.0) { p0_angle_out = p2; return x2_bounds; }
        
        if(fl < 0.0) {
            xl = x1_bounds; xh = x2_bounds;
        } else {
            xh = x1_bounds; xl = x2_bounds;
        }
        
        rts = 0.5*(x1_bounds + x2_bounds);
        dx = std::fabs(x2_bounds - x1_bounds);
        
        for(int j = 0; j < MAXIT; j++) {
            funcd_catenary_local(rts, xacc_tol, f, df, d_geom, l_geom, p0_angle_out);
            
            if(std::fabs(dx) < xacc_tol || std::fabs(f) < xacc_tol) {
                return rts;
            }
            
            // 二分法のみ使用
            dx = 0.5*(xh - xl);
            rts = xl + dx;
            
            if(f < 0.0) {
                xl = rts;
            } else {
                xh = rts;
            }
        }
        
    } catch (...) {
        p0_angle_out = 0.0;
        return 0.5 * (x1_bounds + x2_bounds);
    }
    
    p0_angle_out = 0.0;
    return 0.5 * (x1_bounds + x2_bounds);
}

// ========= カテナリー理論による初期位置設定（簡略版） =========
void ModuleCatenaryLM::InitializeVirtualNodesFromCatenary() {
    if (fairlead_node == nullptr || virtual_nodes.empty()) return;

    const Vec3& fairlead_pos = fairlead_node->GetXCurr();
    Vec3 anchor_pos(APx_orig, APy_orig, APz_orig);
    
    Vec3 FP_AP = fairlead_pos - anchor_pos;
    doublereal h = std::fabs(FP_AP.dGet(3));
    doublereal L_APFP = std::sqrt(std::pow(FP_AP.dGet(1), 2.0) + std::pow(FP_AP.dGet(2), 2.0));

    Vec3 horizontal_dir(1.0, 0.0, 0.0);
    if (L_APFP > 1e-12) {
        horizontal_dir = Vec3(FP_AP.dGet(1) / L_APFP, FP_AP.dGet(2) / L_APFP, 0.0);
    }

    bool use_catenary = false;
    
    // カテナリー計算を試行（簡略版）
    if (h > 1e-6 && L_APFP > 1e-6) {
        try {
            doublereal L0_APFP = L_orig - h;
            doublereal delta = L_APFP - L0_APFP;
            
            if (std::fabs(delta) < L_orig * 0.5) {
                doublereal d = delta / h;
                doublereal l = L_orig / h;
                
                if (l > 1.0 && std::fabs(d) < 2.0) {
                    doublereal x_param, p0 = 0.0;
                    doublereal x1 = 0.1, x2 = 100.0;
                    
                    x_param = rtsafe_catenary_local(x1, x2, xacc_orig, d, l, p0);
                    doublereal H = x_param * w_orig * h;
                    
                    if (H > 1e-6) {
                        doublereal a = H / w_orig;
                        use_catenary = true;
                        
                        for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
                            doublereal s = L_orig * static_cast<doublereal>(Seg_param_minimal - 1 - i) / static_cast<doublereal>(Seg_param_minimal);
                            
                            doublereal x_local = 0.0, z_local = 0.0;
                            
                            doublereal beta = s / a;
                            if (beta < 3.0) {  // 安全な範囲
                                x_local = a * myasinh_local(std::sinh(L_APFP / a) - std::sinh(beta));
                                z_local = a * (std::cosh((L_APFP - x_local) / a) - std::cosh(L_APFP / a));
                            } else {
                                use_catenary = false;
                                break;
                            }
                            
                            virtual_nodes[i].position = anchor_pos + horizontal_dir * x_local + Vec3(0.0, 0.0, z_local);
                            virtual_nodes[i].velocity = Vec3(0.0, 0.0, 0.0);
                        }
                    }
                }
            }
        } catch (...) {
            use_catenary = false;
        }
    }

    // フォールバック：線形補間（元のコードと同じ）
    if (!use_catenary) {
        for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
            doublereal ratio = static_cast<doublereal>(i + 1) / static_cast<doublereal>(Seg_param_minimal);
            virtual_nodes[i].position = fairlead_pos + (anchor_pos - fairlead_pos) * ratio;
            
            // 軽微な垂れ下がり
            doublereal sag = 0.02 * L_orig * ratio * (1.0 - ratio);  // 元の半分以下
            virtual_nodes[i].position += Vec3(0.0, 0.0, -sag);
            
            virtual_nodes[i].velocity = Vec3(0.0, 0.0, 0.0);
        }
    }
}

// ========= 仮想ノード更新（極小版） =========
void ModuleCatenaryLM::UpdateVirtualNodesMinimal(doublereal dCoef) {
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
            doublereal Fz = Kseabed_minimal * pen * 0.1;  // さらに1/10に
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

// ========= 残差ベクトル（極小係留力） =========
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
    UpdateVirtualNodesMinimal(dCoef);

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
            doublereal segment_length = L_orig / static_cast<doublereal>(Seg_param_minimal);
            doublereal strain = (l - segment_length) / segment_length;
            
            if (strain > 0.0) {
                doublereal Fel = EA_minimal * strain * 0.1;  // さらに1/10に
                F_mooring += t * (Fel * fsf);
            }
        }
    }

    R.PutCoef(1, F_mooring.dGet(1));
    R.PutCoef(2, F_mooring.dGet(2));
    R.PutCoef(3, F_mooring.dGet(3));

    return R;
}

// ========= ヤコビアン（極小剛性） =========
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
    doublereal k_minimal = EA_minimal * 0.01;
    K.PutCoef(1, 1, k_minimal);
    K.PutCoef(2, 2, k_minimal);
    K.PutCoef(3, 3, k_minimal);

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
