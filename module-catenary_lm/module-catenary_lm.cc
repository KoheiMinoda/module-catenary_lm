#include "mbconfig.h"           // This goes first in every *.c,*.cc file

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <iomanip>
#include <limits>

#include "dataman.h"
#include "userelem.h"
#include "module-catenary_lm.h" // 自前のヘッダ

// dataman.h や userelem.h から間接的にインクルードされるならいらない可能性
#include "elem.h"          // Elem 基底クラス (userelem.hがインクルードしている可能性が高い)
#include "strnode.h"       // StructNode (ランプドマス点の表現に必須)
#include "drive.h"         // DriveOwner (FSF のため)
#include "node.h"          // Node (pGetNode の戻り値の基底クラス)
#include "gravity.h"

#include <vector>       // std::vector (ノードやセグメントのリスト管理用)

// ========= virtual node =========
struct VirtualNode {
    Vec3 position;
    Vec3 velocity;
    Vec3 acceleration;
    doublereal mass;
    bool active;

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

// クラス宣言
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

    // MBDyn ソルバーがノードの変位や速度を更新した後，要素の内部状態をそれに応じて更新するために呼び出す
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
    double APx_orig; // アンカー点の x 座標
    double APy_orig; // アンカー点の y 座標
    double APz_orig; // アンカー点の z 座標
    double L_orig; // ライン全長
    double w_orig; // チェーンの単重
    double xacc_orig; // rtsafe 関数で使用するパラメータ：収束判定

    DriveOwner FSF_orig; // 元の Force Scale Factor：ランプアップに使う

    unsigned int Seg_param;
    doublereal g_gravity_param;
    // 海底反力の考慮
    doublereal seabed_z_param; // 海底面 Z 座標（上向きが正）
    doublereal Kseabed_param; // 海底ばね [N/m]
    doublereal Cseabed_param; // 海底ダンパ [N s / m]

    StructDispNode* fairlead_node;
    std::vector<VirtualNode> virtual_nodes;

    struct SegmentProperty {
        doublereal L0_seg, M_seg, EA_seg, CA_seg;
        SegmentProperty() : L0_seg(0.0), M_seg(0.0), EA_seg(0.0), CA_seg(0.0) {}
    };
    std::vector<SegmentProperty> P_param;
    std::vector<bool> segSlack_prev;


    // ===== 静的メンバ関数 =====
    // "static" 特定のオブジェクトインスタンスに属さず，クラス自体に関連付けられる
    static double myasinh_local(double val);
    static double myacosh_local(double val);
    static double myatanh_local(double val);
    static void funcd_catenary_local(double x_param, double xacc, double &f_val, double &df_val, double d_geom, double l_geom, double &p0_angle);
    static double rtsafe_catenary_local(double x1_bounds, double x2_bounds, double xacc_tol, double d_geom, double l_geom, double &p0_angle_out);
    // ========================

    void UpdateVirtualNodes(doublereal dCoef, const VectorHandler& XCurr, const VectorHandler& XPrimeCurr);
    void InitializeVirtualNodes();
};

// ============== コンストラクタ：パラメータの読み込みと初期設定 ==========
ModuleCatenaryLM::ModuleCatenaryLM(
    unsigned uLabel,
    const DofOwner *pD0,
    DataManager* pDM,
    MBDynParser& HP
)
    // 初期化子リスト
    : Elem(uLabel, flag(0)), UserDefinedElem(uLabel, pD0),
        // メンバ変数の初期化
        APx_orig(0.0), APy_orig(0.0), APz_orig(0.0),
        L_orig(0.0), w_orig(0.0), xacc_orig(1e-6),
        Seg_param(20),
        g_gravity_param(9.80665),
        fairlead_node(nullptr)
    {
        // 入力ファイルの現在位置が "help" なら使用方法のメッセージを表示
        if (HP.IsKeyWord("help")) {
            silent_cout(
                "\n"
                "Module:     ModuleCatenaryLM (Lumped Mass Catenary Mooring Line)\n"
                "Usage:      catenary_lm, \n"
                "                fairlead_node_label,\n"
                "                total_length, unit_weight, rtsafe_accuracy,\n"
                "                APx, APy, APz, (Anchor fixed coordinates)\n"
                "              [ EA (axial_stiffness), ]\n"
                "              [ CA (axial_damping), ]\n"
                "              [ gravity, g, ]\n"
                "              [ force scale factor, (DriveCaller), ]\n"
                "              [ output, (FLAG) ] ;\n"
                "\n"
                << std::endl
            );
            if (!HP.IsArg()) {
                throw NoErr(MBDYN_EXCEPT_ARGS);
            }
        }

        // ===== パラメータ読み込み開始 =====
        // ここでの読み込みは .mbd ファイルの elements block を上から読み込む（.usr ファイルに外だししているかも）

        // フェアリーダーノード
        unsigned int fairlead_node_label = HP.GetInt();

        // 全長 L_orig
        L_orig = HP.GetReal();
        if (L_orig <= 0.0) {
            throw ErrGeneric(MBDYN_EXCEPT_ARGS);
        }

        // 単位重量 w_orig
        w_orig = HP.GetReal();
        if (w_orig <= 0.0) {
            throw ErrGeneric(MBDYN_EXCEPT_ARGS);
        }

        // rtsafeの計算精度 xacc_orig
        xacc_orig = HP.GetReal();
        if (xacc_orig <= 0.0) {
            throw ErrGeneric(MBDYN_EXCEPT_ARGS);
        }

        // アンカー固定座標 APx_orig, APy_orig, APz_orig
        APx_orig = HP.GetReal();
        APy_orig = HP.GetReal();
        APz_orig = HP.GetReal();

        // ====== EA, CA, g : 既存の .usr ファイルに加える項目？==========
        // 軸剛性 EA (ランプドマス法用)
        doublereal EA_val = 7.536e9; // デフォルト
        if (HP.IsKeyWord("EA")) {
            EA_val = HP.GetReal();
        }
        if (EA_val <= 0.0) {
            throw ErrGeneric(MBDYN_EXCEPT_ARGS);
        }

        // 軸方向減衰係数 CA (オプション)
        doublereal CA_val = 0.0; // デフォルトは減衰なし
        if (HP.IsKeyWord("CA")) {
            CA_val = HP.GetReal();
            if (CA_val < 0.0) {
                throw ErrGeneric(MBDYN_EXCEPT_ARGS);
            }
        }

        // 重力加速度 g (オプション)
        if (HP.IsKeyWord("gravity")) {
            g_gravity_param = HP.GetReal();
            if (g_gravity_param < 0.0) {
                throw ErrGeneric(MBDYN_EXCEPT_ARGS);
            }
        }

        // 海底パラメータ
        seabed_z_param = APz_orig;
        Kseabed_param = 1.e7;
        Cseabed_param = 1.e4;
        if (HP.IsKeyWord("seabed")) {
            if (HP.IsKeyWord("z")) seabed_z_param = HP.GetReal(); // z 座標
            if (HP.IsKeyWord("k")) Kseabed_param = HP.GetReal(); // 剛性 k
            if (HP.IsKeyWord("c")) Cseabed_param = HP.GetReal(); // 減衰 c
        }

        // FSF の読み込み（元のコードを維持）
        if (HP.IsKeyWord("Force" "scale" "factor")) {
			FSF_orig.Set(HP.GetDriveCaller());
		} else {
			FSF_orig.Set(new OneDriveCaller);
		}

        // 出力フラグの設定（元のコードを維持）
        SetOutputFlag(pDM->fReadOutput(HP, Elem::LOADABLE));

        // 各ノードのラベルの受け取り
        // フェアリーダーノードの処理（実ノード）
        {
            Node* rawNode = pDM->pFindNode(Node::STRUCTURAL, fairlead_node_label);
            if (rawNode == nullptr) {
                throw ErrGeneric(MBDYN_EXCEPT_ARGS);
            }
            fairlead_node = dynamic_cast<StructDispNode*>(rawNode);
            if (fairlead_node == nullptr) {
                throw ErrGeneric(MBDYN_EXCEPT_ARGS);
            }
        }

        // 仮想内部ノードの初期化
        virtual_nodes.resize(Seg_param - 1);

        P_param.resize(Seg_param); // Seg_param 個のセグメント
        doublereal L0_s = L_orig / static_cast<doublereal>(Seg_param); // 各セグメントの自然長：単純に全長をセグメント数で割っている
        doublereal mass_per_segment = (w_orig / g_gravity_param) * L0_s; // 各セグメントの質量：各セグメントの「質量」

        // P_param ベクトルの各要素にそれぞれを格納
        for (unsigned int i = 0; i < Seg_param; ++i) {
            P_param[i].L0_seg = L0_s;
            P_param[i].M_seg = mass_per_segment;
            P_param[i].EA_seg = EA_val;
            P_param[i].CA_seg = CA_val;
        }

        // 各仮想ノードに対応する質量
        for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
            doublereal node_mass = 0.0;

            if (i < P_param.size()) {
                node_mass += P_param[i].M_seg * 0.5;
            }
            if (i + 1 < P_param.size()) {
                node_mass += P_param[i + 1].M_seg * 0.5;
            }
            virtual_nodes[i].mass = node_mass;
        }

        segSlack_prev.assign(Seg_param - 1, false);

        InitializeVirtualNodes();

        // デバッグ出力
        for (unsigned int i = 0; i < virtual_nodes.size() && i < 3; ++i) {
            pDM->GetLogFile() << "  Virtual Node " << i << ": " 
                              << virtual_nodes[i].position.dGet(1) << ", "
                              << virtual_nodes[i].position.dGet(2) << ", "
                              << virtual_nodes[i].position.dGet(3) << std::endl;
        }

        // ログ出力：
        pDM->GetLogFile() << "ModuleCatenaryLM (" << GetLabel() << ") initialized:" << std::endl;
        pDM->GetLogFile() << "  Fairlead Node Label: " << fairlead_node_label << std::endl;
        pDM->GetLogFile() << "  Anchor Fixed At: (" << APx_orig << ", " << APy_orig << ", " << APz_orig << ")" << std::endl;
        pDM->GetLogFile() << "  Original Line Length: " << L_orig << ", Unit Weight: " << w_orig << std::endl;
        pDM->GetLogFile() << "  Segments (Fixed): " << Seg_param << std::endl;
        pDM->GetLogFile() << "  EA: " << EA_val << ", CA: " << CA_val << ", Gravity: " << g_gravity_param << std::endl;
        pDM->GetLogFile() << "  RTSAFE Accuracy: " << xacc_orig << std::endl;
    }

ModuleCatenaryLM::~ModuleCatenaryLM(void){}

double ModuleCatenaryLM::myasinh_local(double val) { return std::log(val + std::sqrt(val * val + 1.)); }
double ModuleCatenaryLM::myacosh_local(double val) { return std::log(val + std::sqrt(val + 1.) * std::sqrt(val - 1.));}
double ModuleCatenaryLM::myatanh_local(double val) { return 0.5 * std::log((1. + val) / (1. - val)); }

// ======= 仮想ノードの初期化 =========

void ModuleCatenaryLM::InitializeVirtualNodes() {
    
    if (fairlead_node == nullptr) return;

    const Vec3& fairlead_pos = fairlead_node->GetXCurr();
    Vec3 anchor_pos(APx_orig, APy_orig, APz_orig);

    Vec3 FP_AP = fairlead_pos - anchor_pos;
    doublereal h = std::fabs(FP_AP.dGet(3));
    doublereal L_APFP = std::sqrt(std::pow(FP_AP.dGet(1), 2.0) + std::pow(FP_AP.dGet(2), 2.0));

    Vec3 horizontal_dir(0.0, 0.0, 0.0);
    if (L_APFP > 1e-12) {
        horizontal_dir = Vec3(FP_AP.dGet(1) / L_APFP, FP_AP.dGet(2) / L_APFP, 0.0);
    }

    // 簡単な線形配置から開始（カテナリー計算でエラーが発生する場合のフォールバック）
    bool use_simple_initialization = true;

    try {
        // カテナリー理論による初期位置計算
        doublereal L0_APFP = L_orig - h;
        doublereal delta = L_APFP - L0_APFP;
        doublereal d = 0.0, l = 0.0;

        if (h > 1e-6 && std::fabs(delta) < L_orig * 0.5) {
            d = delta / h;
            l = L_orig / h;

            if (d > 0.0 && l > 1.0 && d < (std::sqrt(l*l - 1.0) - (l - 1.0))) {
                doublereal x_param = 0.0, p0 = 0.0, H = 0.0;
                doublereal x1 = 1e-6, x2 = 1000.0;

                x_param = rtsafe_catenary_local(x1, x2, xacc_orig, d, l, p0);
                H = x_param * w_orig * h;

                if (H > 1e-6) {
                    use_simple_initialization = false;
                    doublereal a = H / w_orig;

                    for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
                        unsigned int seg_idx = i+1;
                        doublereal s = L_orig * (static_cast<doublereal>(Seg_param - seg_idx) / static_cast<doublereal>(Seg_param));

                        doublereal x_local = 0.0, z_local = 0.0;

                        if (p0 < 1e-6) {
                            doublereal beta = s / a;
                            if (beta < 10.0) { // より安全な範囲に制限
                                x_local = a * myasinh_local(std::sinh(L_APFP / a) - std::sinh(beta));
                                z_local = a * (std::cosh((L_APFP - x_local) / a) - std::cosh(L_APFP / a));
                            } else {
                                use_simple_initialization = true;
                                break;
                            }
                        }

                        virtual_nodes[i].position = anchor_pos + horizontal_dir * x_local + Vec3(0.0, 0.0, z_local);
                        virtual_nodes[i].velocity = Vec3(0.0, 0.0, 0.0);
                        virtual_nodes[i].active = true;
                    }
                }
            } 
        }
    } catch (...) {
        use_simple_initialization = true;
    }

    // 簡単な線形補間による初期化（フォールバック）
    if (use_simple_initialization) {
        for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
            doublereal ratio = static_cast<doublereal>(i + 1) / static_cast<doublereal>(Seg_param);
            
            // 線形補間
            virtual_nodes[i].position = fairlead_pos + (anchor_pos - fairlead_pos) * ratio;
            
            // 重力による垂れ下がりを追加
            doublereal sag = 0.1 * L_orig * ratio * (1.0 - ratio); // 簡単な放物線近似
            virtual_nodes[i].position += Vec3(0.0, 0.0, -sag);
            
            // 海底との接触チェック
            if (virtual_nodes[i].position.dGet(3) < seabed_z_param) {
                virtual_nodes[i].position = Vec3(
                    virtual_nodes[i].position.dGet(1), 
                    virtual_nodes[i].position.dGet(2), 
                    seabed_z_param + 0.1 // 少し上に配置
                );
            }
            
            virtual_nodes[i].velocity = Vec3(0.0, 0.0, 0.0);
            virtual_nodes[i].active = true;
        }
    }

}


// =========== 自由度関連 ===================

void ModuleCatenaryLM::WorkSpaceDim(integer* piNumRows, integer* piNumCols) const {

    *piNumRows = 3;
    *piNumCols = 3;
}

void ModuleCatenaryLM::InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const {
    *piNumRows = 3;
    *piNumCols = 3;
}

unsigned int ModuleCatenaryLM::iGetNumConnectedNodes(void) const {
    return 1; // フェアリードのみ
}

unsigned int ModuleCatenaryLM::iGetInitialNumDof(void) const {
    return 0; // 3 かも？
}

const Node* ModuleCatenaryLM::pGetNode(unsigned int i) const {
    if (i == 1) {
        return fairlead_node;
    }
    return nullptr;
}

// 非 const 版：ノードの状態を変更することが可能
Node* ModuleCatenaryLM::pGetNode(unsigned int i) {
    if (i == 1) {
        return fairlead_node;
    }
    return nullptr;
}

// ========= 具体的な初期化 ===========
void ModuleCatenaryLM::SetInitialValue(VectorHandler& X, VectorHandler& XP) {
    InitializeVirtualNodes();
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
    const VectorHandler& /*X*/
) {
    WorkMat.SetNullMatrix();
    return WorkMat;
}

// ======= AssRes ========

SubVectorHandler& ModuleCatenaryLM::AssRes(
    SubVectorHandler& R,
    doublereal dCoef,
    const VectorHandler& XCurr,
    const VectorHandler& XPrimeCurr
) {
    
    if (fairlead_node == nullptr) {
        silent_cerr("ModuleCatenaryLM(" << GetLabel() << "): "
                   "fairlead node not available" << std::endl);
        throw ErrGeneric(MBDYN_EXCEPT_ARGS);
    }

    const doublereal fsf = FSF_orig.dGet();

    // フェアリーダーノードの3自由度のみ
    R.ResizeReset(3);

    // フェアリーダーノードのインデックス設定
    integer iFirstIndex = fairlead_node->iGetFirstPositionIndex();
    R.PutRowIndex(1, iFirstIndex + 1);
    R.PutRowIndex(2, iFirstIndex + 2);
    R.PutRowIndex(3, iFirstIndex + 3);

    // 仮想ノードの状態更新
    UpdateVirtualNodes(dCoef, XCurr, XPrimeCurr);

    // フェアリーダーノードに作用する力の計算
    const Vec3& fairlead_pos = fairlead_node->GetXCurr();
    const Vec3& fairlead_vel = fairlead_node->GetVCurr();

    // 最初の仮想ノードとの間の軸力
    Vec3 F_mooring(0.0, 0.0, 0.0);
    
    if (virtual_nodes.size() > 0 && virtual_nodes[0].active) {
        Vec3 dx = virtual_nodes[0].position - fairlead_pos;
        doublereal l = dx.Norm();
        
        if (l > 1e-12) {
            Vec3 t = dx / l;
            
            doublereal L0 = P_param[0].L0_seg;
            doublereal EA = P_param[0].EA_seg;
            doublereal CA = P_param[0].CA_seg;
            
            doublereal Fel = EA * (l - L0) / L0;
            Vec3 dv = virtual_nodes[0].velocity - fairlead_vel;
            doublereal vrel = dv(1)*t(1) + dv(2)*t(2) + dv(3)*t(3);
            doublereal Fd = vrel * CA;
            
            doublereal Fax = Fel + Fd;
            if (Fax < 0.0) Fax = 0.0;
            
            F_mooring = t * (Fax * fsf);
        }
    }

    // フェアリーダーノードに係留力を適用
    R.PutCoef(1, F_mooring.dGet(1));
    R.PutCoef(2, F_mooring.dGet(2));
    R.PutCoef(3, F_mooring.dGet(3));

    return R;
}

// ========= AssJac =========

VariableSubMatrixHandler& ModuleCatenaryLM::AssJac(
    VariableSubMatrixHandler& WH,
    doublereal dCoef,
    const VectorHandler& XCurr, 
    const VectorHandler& XPrimeCurr
) {
    if (fairlead_node == nullptr) {
        silent_cerr("ModuleCatenaryLM(" << GetLabel() << "): "
                   "fairlead node not available" << std::endl);
        throw ErrGeneric(MBDYN_EXCEPT_ARGS);
    }

    // フェアリーダーノードの3x3剛性行列
    FullSubMatrixHandler& K = WH.SetFull();
    K.ResizeReset(3, 3);

    // フェアリーダーノードのインデックス設定
    integer iFirstIndex = fairlead_node->iGetFirstPositionIndex();
    for (int i = 0; i < 3; ++i) {
        K.PutRowIndex(i + 1, iFirstIndex + i + 1);
        K.PutColIndex(i + 1, iFirstIndex + i + 1);
    }

    // 係留力の剛性計算
    if (virtual_nodes.size() > 0 && virtual_nodes[0].active) {
        const Vec3& fairlead_pos = fairlead_node->GetXCurr();
        Vec3 dx = virtual_nodes[0].position - fairlead_pos;
        doublereal l = dx.Norm();
        
        if (l > 1e-12) {
            Vec3 t = dx / l;
            
            doublereal EA = P_param[0].EA_seg;
            doublereal CA = P_param[0].CA_seg;
            doublereal L0 = P_param[0].L0_seg;
            
            doublereal k_ax = EA / L0;
            doublereal c_ax = -CA * dCoef;
            
            doublereal Fel = EA * (l - L0) / L0;
            if (Fel > 0.0) {
                // 軸方向剛性と幾何剛性
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        doublereal K_tt = (k_ax + c_ax) * t(r+1) * t(c+1);
                        doublereal delta = (r == c) ? 1.0 : 0.0;
                        doublereal Kg = (Fel / l) * (delta - t(r+1) * t(c+1));
                        
                        K.PutCoef(r + 1, c + 1, -(K_tt + Kg)); // 負号：係留力は復元力
                    }
                }
            }
        }
    }

    return WH;
}


// rtsafe_catenary_local 関数関数から呼び出される：f(x) とその導関数を求める
void ModuleCatenaryLM::funcd_catenary_local(double x_param, double xacc, double &f_val, double &df_val, double d_geom, double l_geom, double &p0_angle) {
    int i_internal_iter, max_internal_iter;
    double f1_internal, df1_internal;
    max_internal_iter = 1000;

    // 水平張力が無い状態
    if(x_param == 0.0) {
        f_val = -d_geom;
        df_val = 0.0;
        p0_angle = 0.0;

    // 水平張力がある状態
    } else if(x_param > 0.0) {
        
        // 特殊なケース：元のコードをそのままにしつつ，保護をすこししただけ
        if(l_geom <= 0.0){
            double X_1_internal;
            X_1_internal = 1.0/x_param+1.0;
            if (X_1_internal < 1.0) X_1_internal = 1.0; // acoshの引数保護
                double term_sqrt1 = 1.0+2.0*x_param;
            if (term_sqrt1 < 0.0) term_sqrt1 = 0.0;
                double term_sqrt2 = X_1_internal*X_1_internal-1.0;
            if (term_sqrt2 < 0.0) term_sqrt2 = 0.0;

            f_val=x_param*myacosh_local(X_1_internal)-std::sqrt(term_sqrt1)+1.0-d_geom;
            if (std::fabs(x_param * std::sqrt(term_sqrt2)) < 1e-12 || std::fabs(std::sqrt(term_sqrt1)) < 1e-12 ) { // ゼロ除算保護
                df_val = myacosh_local(X_1_internal); // 近似
            } else {
                df_val=myacosh_local(X_1_internal)-1.0/std::sqrt(term_sqrt1)-1.0/(x_param*std::sqrt(term_sqrt2));
            }
            p0_angle=0.0;

        // 一般的なケース
        } else {

            // 海底との接触がある可能性がある場合：元のコードをそのままにしつつ，保護をすこししただけ
            if(x_param > (l_geom*l_geom - 1.0) / 2.0) {
                p0_angle=0.0;
                for(i_internal_iter = 0; i_internal_iter < max_internal_iter; i_internal_iter++) {
                    double cos_p0 = std::cos(p0_angle);
                    if (std::fabs(cos_p0) < 1e-9) { df1_internal = 1.0; break; } // 保護
                    double func1_internal = 1.0/x_param + 1.0/cos_p0;
                    double term_in_sqrt_f1 = func1_internal*func1_internal - 1.0;
                    if (term_in_sqrt_f1 < 0.0) term_in_sqrt_f1 = 0.0;

                    f1_internal = x_param*(std::sqrt(term_in_sqrt_f1) - std::tan(p0_angle)) - l_geom;

                    if (std::fabs(cos_p0) < 1e-9 || term_in_sqrt_f1 < 1e-12 ) { df1_internal = 1.0; break; } // 保護
                    df1_internal = x_param * (func1_internal * std::tan(p0_angle) / (cos_p0 * std::sqrt(term_in_sqrt_f1)) - (std::tan(p0_angle)*std::tan(p0_angle)) - 1.0);

                    if (std::fabs(df1_internal) < 1e-9) { break; }
                    p0_angle = p0_angle-f1_internal/df1_internal;

                    cos_p0 = std::cos(p0_angle);
                    if (std::fabs(cos_p0) < 1e-9) { break; }
                    func1_internal = 1.0/x_param + 1.0/cos_p0;
                    term_in_sqrt_f1 = func1_internal*func1_internal - 1.0;
                    if (term_in_sqrt_f1 < 0.0) term_in_sqrt_f1 = 0.0;
                    f1_internal = x_param*(std::sqrt(term_in_sqrt_f1) - std::tan(p0_angle)) - l_geom;

                    if(std::fabs(f1_internal) < xacc) { break; }
                }
                if(i_internal_iter == max_internal_iter && std::fabs(f1_internal) > xacc) {
                }

                double X_2_internal = l_geom/x_param + std::tan(p0_angle);
                double X_3_internal = std::tan(p0_angle);
                f_val = x_param*(myasinh_local(X_2_internal) - myasinh_local(X_3_internal)) - l_geom + 1.0 - d_geom;
                
                double term_in_sqrt_df = X_2_internal*X_2_internal + 1.0;
                // if (term_in_sqrt_df < 0.0) term_in_sqrt_df = 0.0; // asinhの引数は実数なので常に正
                if (std::fabs(x_param * std::sqrt(term_in_sqrt_df)) < 1e-12 ) { df_val = 1.0; } // 保護
                else {
                    df_val=myasinh_local(X_2_internal) - myasinh_local(X_3_internal) - l_geom/(x_param*std::sqrt(term_in_sqrt_df));
                }

            // 海底との接触がある可能性が無い場合：単純なカテナリー
            } else {
                double X_5_internal = 1.0/x_param+1.0;
                if (X_5_internal < 1.0) X_5_internal = 1.0; // acoshの引数保護
                double term_sqrt1 = 1.0+2.0*x_param;
                if (term_sqrt1 < 0.0) term_sqrt1 = 0.0;
                double term_sqrt2 = X_5_internal*X_5_internal-1.0;
                if (term_sqrt2 < 0.0) term_sqrt2 = 0.0;

                f_val = x_param*myacosh_local(X_5_internal) - std::sqrt(term_sqrt1) + 1.0 - d_geom;
                if (std::fabs(x_param * std::sqrt(term_sqrt2)) < 1e-12 || std::fabs(std::sqrt(term_sqrt1)) < 1e-12) { // 保護
                     df_val = myacosh_local(X_5_internal); // 近似
                } else {
                     df_val = myacosh_local(X_5_internal) - 1.0/std::sqrt(term_sqrt1) - 1.0/(x_param*std::sqrt(term_sqrt2));
                }
                p0_angle = 0.0;
            }
        }
    } else {
        throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
    }
}

// x1_bounds, x2_bounds は根が含まれると期待される初期区間の下限と上限
double ModuleCatenaryLM::rtsafe_catenary_local(double x1_bounds, double x2_bounds, double xacc_tol, double d_geom, double l_geom, double &p0_angle_out) {
    const int MAXIT_internal = 1000;
    int j_internal;
    double fh_internal,fl_internal,xh_internal,xl_internal;
    double dx_internal,dxold_internal,f_internal,temp_internal,rts_internal;
    double p1_internal, p2_internal;
    double df_not_used, df_internal; // funcd_catenary_local がdfを要求するため

    // x1_bounds, x2_bounds における f(x) とその導関数
    funcd_catenary_local(x1_bounds, xacc_tol, fl_internal, df_not_used, d_geom, l_geom, p1_internal);
    funcd_catenary_local(x2_bounds, xacc_tol, fh_internal, df_not_used, d_geom, l_geom, p2_internal);

    // 二つの関数値が同符号ならエラー
    if((fl_internal > 0.0 && fh_internal > 0.0) || (fl_internal < 0.0 && fh_internal < 0.0)) {
        throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
    }
    // どちらかの関数値が 0 であればそれが答え
    if(fl_internal == 0.0) {
        p0_angle_out = p1_internal;
        return x1_bounds;
    }
    if(fh_internal == 0.0) {
        p0_angle_out = p2_internal;
        return x2_bounds;
    }

    // fl_internal < 0 となるように xl_internal と xh_internal を設定する
    if(fl_internal < 0.0) {
        xl_internal = x1_bounds;
        xh_internal = x2_bounds;
    } else {
        xh_internal = x1_bounds;
        xl_internal = x2_bounds;
    }

    // 中点を初期の推定値とし，反復処理のセット
    rts_internal = 0.5*(x1_bounds + x2_bounds);
    dxold_internal = std::fabs(x2_bounds - x1_bounds);
    dx_internal = dxold_internal;
    funcd_catenary_local(rts_internal, xacc_tol, f_internal, df_internal, d_geom, l_geom, p0_angle_out);


    for(j_internal = 0; j_internal < MAXIT_internal; j_internal++) {
        if((((rts_internal - xh_internal)*df_internal - f_internal)*((rts_internal - xl_internal)*df_internal - f_internal) > 0.0)
           || (std::fabs(2.0*f_internal) > std::fabs(dxold_internal*df_internal))) {
            dxold_internal = dx_internal;
            dx_internal = 0.5*(xh_internal - xl_internal);
            rts_internal = xl_internal + dx_internal;
            if(xl_internal == rts_internal) { return rts_internal; }
        } else {
            dxold_internal = dx_internal;
            if (std::fabs(df_internal) < 1e-12) { // ゼロ除算を避ける
                dx_internal = 0.5*(xh_internal - xl_internal); // 二分法にフォールバック
                rts_internal = xl_internal + dx_internal;
                if(xl_internal == rts_internal){ return rts_internal; }
            } else {
                dx_internal = f_internal/df_internal;
            }
            temp_internal = rts_internal;
            rts_internal -= dx_internal;
            if(temp_internal == rts_internal) {return rts_internal;}
        }

        if(std::fabs(dx_internal) < xacc_tol) { return rts_internal; }

        funcd_catenary_local(rts_internal, xacc_tol, f_internal, df_internal, d_geom, l_geom, p0_angle_out);

        if(f_internal < 0.0){
            xl_internal = rts_internal;
        } else {
            xh_internal = rts_internal;
        }
    }

    throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
}

// Outputメソッド: シミュレーション結果の出力
void ModuleCatenaryLM::Output(OutputHandler& OH) const {
    if (!bToBeOutput()) return;
    if (!OH.UseText(OutputHandler::LOADABLE)) return;

    const integer lbl = GetLabel();

    // フェアリーダ座標
    if (fairlead_node == nullptr) {
        OH.Loadable() << lbl << "[Error] Fairlead node not available for output.\n";
        return;
    }
    const Vec3 posFL = fairlead_node -> GetXCurr();

    // フェアリーダに作用する軸方向力
    Vec3 Ffl(0.0, 0.0, 0.0);
    if (virtual_nodes.size() > 0 && virtual_nodes[0].active) {
        Vec3 dx = virtual_nodes[0].position - posFL;
        doublereal l = dx.Norm();
        if (l > 1e-12) {
            Vec3 t = dx / l;
            
            doublereal L0 = P_param[0].L0_seg;
            doublereal EA = P_param[0].EA_seg;
            
            doublereal Fel = EA * (l - L0) / L0;
            if (Fel > 0.0) {
                const doublereal fsf = FSF_orig.dGet();
                Ffl = t * (Fel * fsf);
            }
        }
    }
    
    OH.Loadable() << lbl << " FairleadPos "
                  << posFL.dGet(1) << " "
                  << posFL.dGet(2) << " "
                  << posFL.dGet(3) << " FairleadForce "
                  << Ffl.dGet(1)  << " "
                  << Ffl.dGet(2)  << " "
                  << Ffl.dGet(3);

    /* // 仮想ノードの位置も出力する場合

    OH.Loadable() << " VirtualNodes " << virtual_nodes.size();
    for (unsigned int i = 0; i < virtual_nodes.size() && i < 5; ++i) { // 最初の5個のみ
        if (virtual_nodes[i].active) {
            OH.Loadable() << " " << virtual_nodes[i].position.dGet(1)
                         << " " << virtual_nodes[i].position.dGet(2)
                         << " " << virtual_nodes[i].position.dGet(3);
        }
    }
    
    */
    
    OH.Loadable() << '\n';
}


// =========== その他の関数 必要に応じて編集する ============
// 後回し
std::ostream& ModuleCatenaryLM::Restart(std::ostream& out) const {
    out << "# ModuleCatenaryLM (Label: " << GetLabel() << ") Restart: Not implemented yet." << std::endl;
    return out;
}

void ModuleCatenaryLM::SetValue(
    DataManager* /*pDM*/,
    VectorHandler& /*X*/,
    VectorHandler& /*XP*/,
    SimulationEntity::Hints* /*pHints*/ // 線形化が必要かどうか
)
{
    // 重力荷重は AssRes 内で扱うか，要素外で扱うのでここは空実装
}

// ========= 内部ノードのアップデート ========

void ModuleCatenaryLM::UpdateVirtualNodes(doublereal dCoef, const VectorHandler& XCurr, const VectorHandler& XPrimeCurr) {
    
    if (fairlead_node == nullptr) return;

    // フェアリーダーの現在状態を取得
    const Vec3& fairlead_pos = fairlead_node->GetXCurr();
    const Vec3& fairlead_vel = fairlead_node->GetVCurr();

    Vec3 anchor_pos(APx_orig, APy_orig, APz_orig);

    const doublereal dt = 1.0 / std::max(std::fabs(dCoef), 1e-6);
    const doublereal max_dt = 0.001;
    const doublereal actual_dt = std::min(dt, max_dt);

    const int sub_steps = std::max(1, static_cast<int>(actual_dt / 0.0005));
    const doublereal sub_dt = actual_dt / sub_steps;

    for (int sub_step = 0; sub_step < sub_steps; ++sub_step) {
        for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
            if (!virtual_nodes[i].active) continue;

            // 重力
            Vec3 F_gravity(0.0, 0.0, -virtual_nodes[i].mass * g_gravity_param);

            // 海底反力（より安定な実装）
            Vec3 F_seabed(0.0, 0.0, 0.0);
            if (virtual_nodes[i].position.dGet(3) < seabed_z_param) {
                doublereal pen = std::max(0.0, seabed_z_param - virtual_nodes[i].position.dGet(3));
                doublereal vz = virtual_nodes[i].velocity.dGet(3);
                
                // 海底反力を制限して数値不安定を防ぐ
                doublereal Fz = std::min(Kseabed_param * pen - Cseabed_param * vz, 
                                       virtual_nodes[i].mass * g_gravity_param * 10.0); // 重力の10倍まで
                F_seabed = Vec3(0.0, 0.0, std::max(0.0, Fz)); // 下向きの力は制限
            }

            // 隣接ノードとの軸力計算
            Vec3 F_axial(0.0, 0.0, 0.0);

            // 左隣（フェアリーダーまたは他の仮想ノード）
            Vec3 left_pos, left_vel;
            if (i == 0) {
                left_pos = fairlead_pos;
                left_vel = fairlead_vel;
            } else {
                left_pos = virtual_nodes[i - 1].position;
                left_vel = virtual_nodes[i - 1].velocity;
            }

            Vec3 dx_left = virtual_nodes[i].position - left_pos;
            doublereal l_left = dx_left.Norm();

            if (l_left > 1e-12) {
                Vec3 t_left = dx_left / l_left;

                unsigned int seg_idx_left = (i < P_param.size()) ? i : (P_param.size() - 1);

                doublereal L0 = P_param[seg_idx_left].L0_seg;
                doublereal EA = P_param[seg_idx_left].EA_seg;
                doublereal CA = P_param[seg_idx_left].CA_seg;

                // ひずみを制限して大変形を防ぐ
                doublereal strain = (l_left - L0) / L0;
                strain = std::max(-0.5, std::min(strain, 2.0)); // ひずみを±50%〜200%に制限
                
                doublereal Fel = EA * strain;
                Vec3 dv = virtual_nodes[i].velocity - left_vel;
                doublereal vrel = dv(1)*t_left(1) + dv(2)*t_left(2) + dv(3)*t_left(3);
                doublereal Fd = vrel * CA;

                doublereal Fax = Fel + Fd;
                
                // 軸力の上限を設定（質量に比例した合理的な値）
                doublereal max_force = virtual_nodes[i].mass * g_gravity_param * 100.0; // 重力の100倍まで
                Fax = std::max(-max_force, std::min(Fax, max_force));
                
                if (Fax < 0.0) Fax = 0.0; // 圧縮力は無効

                F_axial += t_left * (-Fax); // 引張方向
            }

            // 右隣（アンカーまたは他の仮想ノード）
            Vec3 right_pos;
            Vec3 right_vel(0.0, 0.0, 0.0);
            if (i == virtual_nodes.size() - 1) {
                right_pos = Vec3(APx_orig, APy_orig, APz_orig); // アンカー
                right_vel = Vec3(0.0, 0.0, 0.0); // アンカーは固定
            } else {
                right_pos = virtual_nodes[i + 1].position;
                right_vel = virtual_nodes[i + 1].velocity;
            }

            Vec3 dx_right = right_pos - virtual_nodes[i].position;
            doublereal l_right = dx_right.Norm();
            if (l_right > 1e-12) {
                Vec3 t_right = dx_right / l_right;
                unsigned int seg_idx_right = (i + 1 < P_param.size()) ? (i + 1) : (P_param.size() - 1);
                
                doublereal L0 = P_param[seg_idx_right].L0_seg;
                doublereal EA = P_param[seg_idx_right].EA_seg;
                doublereal CA = P_param[seg_idx_right].CA_seg;

                // ひずみを制限して大変形を防ぐ
                doublereal strain = (l_right - L0) / L0;
                strain = std::max(-0.5, std::min(strain, 2.0)); // ひずみを±50%〜200%に制限
                
                doublereal Fel = EA * strain;
                Vec3 dv = right_vel - virtual_nodes[i].velocity;
                doublereal vrel = dv(1)*t_right(1) + dv(2)*t_right(2) + dv(3)*t_right(3);
                doublereal Fd = vrel * CA;

                doublereal Fax = Fel + Fd;
                
                // 軸力の上限を設定
                doublereal max_force = virtual_nodes[i].mass * g_gravity_param * 100.0;
                Fax = std::max(-max_force, std::min(Fax, max_force));
                
                if (Fax < 0.0) Fax = 0.0; // 圧縮力は無効

                F_axial += t_right * Fax;
            }

            Vec3 F_total = F_gravity + F_seabed + F_axial;

            // 加速度計算
            if (virtual_nodes[i].mass > 1e-12) {
                virtual_nodes[i].acceleration = F_total / virtual_nodes[i].mass;
                
                // 加速度を制限（急激な変化を防ぐ）
                doublereal acc_norm = virtual_nodes[i].acceleration.Norm();
                doublereal max_acc = 100.0 * g_gravity_param; // 重力加速度の100倍まで
                if (acc_norm > max_acc) {
                    virtual_nodes[i].acceleration = virtual_nodes[i].acceleration * (max_acc / acc_norm);
                }
            } else {
                virtual_nodes[i].acceleration = Vec3(0.0, 0.0, 0.0);
            }

            // 速度更新（減衰を追加して安定性向上）
            doublereal damping_factor = 0.99; // 軽微な数値減衰
            virtual_nodes[i].velocity = virtual_nodes[i].velocity * damping_factor + 
                                      virtual_nodes[i].acceleration * sub_dt;
            
            // 速度制限
            doublereal vel_norm = virtual_nodes[i].velocity.Norm();
            doublereal max_vel = 50.0; // 最大速度 50 m/s
            if (vel_norm > max_vel) {
                virtual_nodes[i].velocity = virtual_nodes[i].velocity * (max_vel / vel_norm);
            }

            // 位置更新
            Vec3 old_position = virtual_nodes[i].position;
            virtual_nodes[i].position += virtual_nodes[i].velocity * sub_dt;
            
            // 位置の妥当性チェック（急激な移動を防ぐ）
            Vec3 displacement = virtual_nodes[i].position - old_position;
            doublereal disp_norm = displacement.Norm();
            doublereal max_disp = L_orig * 0.1; // 全長の10%まで
            if (disp_norm > max_disp) {
                virtual_nodes[i].position = old_position + displacement * (max_disp / disp_norm);
                // 速度も調整
                virtual_nodes[i].velocity = virtual_nodes[i].velocity * 0.5;
            }
            
            // 海底貫入の防止（硬い制約）
            if (virtual_nodes[i].position.dGet(3) < seabed_z_param) {
                virtual_nodes[i].position = Vec3(virtual_nodes[i].position.dGet(1), 
                                                virtual_nodes[i].position.dGet(2), 
                                                seabed_z_param);
                // Z方向速度をゼロにする
                if (virtual_nodes[i].velocity.dGet(3) < 0.0) {
                    virtual_nodes[i].velocity = Vec3(virtual_nodes[i].velocity.dGet(1),
                                                   virtual_nodes[i].velocity.dGet(2),
                                                   0.0);
                }
            }
        }
    }
}

// ========== モジュール登録関数 ============

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
