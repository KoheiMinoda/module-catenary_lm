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

// クラス宣言
class ModuleCatenaryLM : virtual public Elem, public UserDefinedElem
{
public:
    // コンストラクタ [uLabel : 要素のラベル, *pD0 : DOF(Degree of Freedom) 所有者へのポインタ, *pDM : データマネージャーへのポインタ, HP : MBDyn パーサへの"参照"] 
    ModuleCatenaryLM(unsigned uLabel, const DofOwner *pD0, DataManager* pDM, MBDynParser& HP);
    // デストラクタ [オブジェクトが破棄されるたび呼びだされる] // メモリリークを防ぐために重要
    virtual ~ModuleCatenaryLM(void);

    // ========= 仮想関数群 ここから ==============
    // クラス宣言の際に Elem と UserDefinedElem を継承している
    // 以下の "virtual" 宣言は (Elem と UserDefinedElem) で宣言されている仮想関数を再定義することを意味する
    // "const" のキーワードがついているメンバ関数は，その関数内でオブジェクトのメンバ変数を変更しないという宣言

    // シミュレーション中に特定の要素の状態（FP 点の位置や張力など）を指定された形式で出力ファイルに書き出すために MBDyn から呼び出される
    virtual void Output(OutputHandler& OH) const;
    // ヤコビアン行列，残差行列のサイズを決める：ランプドマスモデルでは，関連するノードの自由度に依存
    virtual void WorkSpaceDim(integer* piNumRows, integer* piNumCols) const;

    // 要素の運動方程式を現在の状態変数とその時間微分で偏微分下ヤコビアン行列の"成分"を計算し，MBDyn の全体ヤコビアン行列の対応する部分に格納する
    VariableSubMatrixHandler&
    AssJac(VariableSubMatrixHandler& WorkMat, 
        doublereal dCoef, 
        const VectorHandler& XCurr, 
        const VectorHandler& XPrimeCurr
    );

    // 要素の運動方程式の外力を計算する
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
    ) override; // override は C++11 以降で基底クラスに仮想関数がなければコンパイルエラーを発生させるという機能：無くても良い

    // この要素が直接族または管理しているノードの数を返す
    virtual unsigned int iGetNumConnectedNodes(void) const;
    // シミュレーション開始時に要素の初期状態や初期ノードの初期値を設定する
    virtual void SetInitialValue(VectorHandler& X, VectorHandler& XP);
    // 中断した後に再開する場合に内部情報を保存する
    virtual std::ostream& Restart(std::ostream& out) const;

    virtual unsigned int iGetInitialNumDof(void) const;
    virtual void InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const;
    virtual VariableSubMatrixHandler&
    InitialAssJac(VariableSubMatrixHandler& WorkMat, const VectorHandler& XCurr);
    virtual SubVectorHandler&
    InitialAssRes(SubVectorHandler& WorkVec, const VectorHandler& XCurr);

    virtual const Node* pGetNode(unsigned int i) const;
    virtual Node* pGetNode(unsigned int i);
    // ========= 仮想関数群 ここまで ==============

private:
    double APx_orig; // アンカー点の x 座標
    double APy_orig; // アンカー点の y 座標
    double APz_orig; // アンカー点の z 座標
    double L_orig; // ライン全長
    double w_orig; // チェーンの単重
    double xacc_orig; // rtsafe 関数で使用するパラメータ：収束判定

    DriveOwner FSF_orig; // 元の Force Scale Factor：ランプアップに使う

    // ===== 静的メンバ関数 =====
    // "static" 特定のオブジェクトインスタンスに属さず，クラス自体に関連付けられる
    static double myasinh_local(double val);
    static double myacosh_local(double val);
    static double myatanh_local(double val);
    static void funcd_catenary_local(double x_param, double xacc, double &f_val, double &df_val, double d_geom, double l_geom, double &p0_angle);
    static double rtsafe_catenary_local(double x1_bounds, double x2_bounds, double xacc_tol, double d_geom, double l_geom, double &p0_angle_out);
    // ========================

    unsigned int Seg_param;

    // ノード配列 (StructNode* の動的配列)
    // N_nodes_param[0] はフェアリーダーノード
    // N_nodes_param[1] ... N_nodes_param[Seg_param-1] は内部質量点ノード
    // アンカーは固定座標(APx_orig, APy_orig, APz_orig)として別途扱う
    // このベクトルサイズは Seg_param になる（0 ~ Seg_param - 1）
    std::vector<StructDispNode*> N_nodes_param;

    std::vector<doublereal> node_mass_param; // 各ランプドマス点の質量 [kg]

    // 各セグメントの特性
    struct SegmentProperty {
        doublereal L0_seg;  // 「セグメント」の初期自然長
        doublereal M_seg;   // 「セグメント」の質量
        doublereal EA_seg;  // 「セグメント」の軸剛性 (EA)
        doublereal CA_seg;  // 「セグメント」の軸方向減衰係数 (オプション)

        SegmentProperty() : L0_seg(0.0), M_seg(0.0), EA_seg(0.0), CA_seg(0.0) {}
    };
    std::vector<SegmentProperty> P_param; // サイズは Seg_param (セグメント数)

    doublereal g_gravity_param;

    // 海底反力の考慮
    doublereal seabed_z_param; // 海底面 Z 座標（上向きが正）
    doublereal Kseabed_param; // 海底ばね [N/m]
    doublereal Cseabed_param; // 海底ダンパ [N s / m]

    std::vector<bool> segSlack_prev; // 前回計算時のスラック状態
    doublereal Krot_lock_param; // 回転 DOF を縛るペナルティ剛性
};

// コンストラクタ：パラメータの読み込みと初期設定
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
        g_gravity_param(9.80665)
    {
        // 入力ファイルの現在位置が "help" なら使用方法のメッセージを表示
        if (HP.IsKeyWord("help")) {
            silent_cout(
                "\n"
                "Module:     ModuleCatenaryLM (Lumped Mass Catenary Mooring Line - Segments Fixed to 20)\n"
                "Usage:      catenary_lm, \n"
                "                fairlead_node_label,\n"
                "                total_length, unit_weight, rtsafe_accuracy,\n"
                "                APx, APy, APz, (Anchor fixed coordinates)\n"
                "                EA (axial_stiffness),\n"
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
        doublereal EA_val;
        if (HP.IsKeyWord("EA")) { // キーワードがある方がより頑健
            EA_val = HP.GetReal();
        } else {
            EA_val = HP.GetReal(); // キーワードなしで順番に読む場合
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
            if (g_gravity_param < 0.0) { // 正の値
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

        // ノードの確保と初期化
        // N_nodes_param にはフェアリーダーと Seg_param-1 個の内部ノードを格納
        // 合計 Seg_param 個の StructNode* を持つ
        N_nodes_param.clear();
        N_nodes_param.resize(Seg_param);

        // ====== ここも既存の .usr ファイルに書き足す ======
        unsigned int fairlead_node_label = HP.GetInt();

        // 各ノードのラベルの受け取り
        // フェアリーダーノードの処理
        {
            Node* rawNode = pDM->pFindNode(Node::STRUCTURAL, fairlead_node_label);
            if (rawNode == nullptr) {
                throw ErrGeneric(MBDYN_EXCEPT_ARGS);
            }
            StructDispNode* dispNode = dynamic_cast<StructDispNode*>(rawNode);
            if (dispNode == nullptr) {
                throw ErrGeneric(MBDYN_EXCEPT_ARGS);
            }
            N_nodes_param[0] = dispNode;
        }

        /*
        // .ref で定義せずに，各内部ノードのデータは追わない場合
        const integer BASE_LBL = 90000;
        for (unsigned int i = 1; i < Seg_param; ++i) {

            integer lbl = BASE_LBL + i;

            StructDispNode* pNew = new StructDispNode(
                nextLbl + 1,
                *pD0,
                Vec3(0., 0., 0.),
                Mat3x3DEye,
                Vec3(0., 0., 0.),
                Vec3(0., 0., 0.)
            );

            pDM -> AddNode(pNew);
            N_nodes_param[i] = pNew;
        }
        */

        // ====== 各ノードの時系列データを追うならこれで良いはず？ ============
        // 一旦は FP だけにして後回し
        // 内部質量点ノードの処理（1 … Seg_param-1）もすべて入力ファイルでラベルを受け取り
        for (unsigned int i = 1; i < Seg_param; ++i) {
            unsigned int node_label = HP.GetInt();

            Node* rawNode = pDM->pFindNode(Node::STRUCTURAL, node_label);
            if (rawNode == nullptr) {
                throw ErrGeneric(MBDYN_EXCEPT_ARGS);
            }
            StructDispNode* dispNode = dynamic_cast<StructDispNode*>(rawNode);
            if (dispNode == nullptr) {
                throw ErrGeneric(MBDYN_EXCEPT_ARGS);
            }
            N_nodes_param[i] = dispNode;
        }

        Vec3 zeroPos(0.0, 0.0, 0.0);
        Vec3 zeroVel(0.0, 0.0, 0.0);

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

        // 各ノードに対応する質量
        node_mass_param.assign(Seg_param, 0.0);
        for (unsigned int s = 0; s < Seg_param; ++s) {
            const doublereal m_half = P_param[s].M_seg*0.5;
            node_mass_param[s] += m_half; // 右端側
            if (s+1 < Seg_param) {
                node_mass_param[s+1] += m_half; // 左隣側
            }
        }

        // 各ノードに対応する慣性の情報をもたせる？
        // 具体的に慣性を足すのは AssRes 内部

        Krot_lock_param = 1.e12;
        segSlack_prev.assign(Seg_param - 1, false);

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

// 要素に関連する自由度の初期値を設定するために MBDyn から呼び出す
// N_nodes_params に格納されているノードの初期位置や初期速度を計算
// フェアリーダー点座標とアンカー点座標からラインの初期形状を計算し，各内部ノードの初期座標を決定する
void ModuleCatenaryLM::SetInitialValue(VectorHandler& X, VectorHandler& XP) {
    
    const Vec3& fairlead_pos = N_nodes_param[0] -> GetXCurr(); // フェアリーダーノードの初期位置を取得
    Vec3 anchor_pos(APx_orig, APy_orig, APz_orig); // アンカーの位置を取得

    Vec3 FP_AP = fairlead_pos - anchor_pos; // アンカーを原点とした座標系でのフェアリーダーの位置ベクトル

    doublereal h = std::fabs(FP_AP.dGet(3)); // FP, AP の鉛直距離の計算
    doublereal L_APFP = std::sqrt(std::pow(FP_AP.dGet(1), 2.0) + std::pow(FP_AP.dGet(2), 2.0)); // FP, AP の水平距離の計算

    // 水平面で見て，AP から FP に向かう単位ベクトル
    Vec3 horizontal_dir(0.0, 0.0, 0.0);
    if (L_APFP > 1e-12) {
        horizontal_dir = Vec3(FP_AP.dGet(1) / L_APFP, FP_AP.dGet(2) / L_APFP, 0.0);
    }

    // カテナリー理論に用いるパラメータ
    doublereal L0_APFP = L_orig - h; // 水平張力が 0 になるときの水平距離
    doublereal delta = L_APFP - L0_APFP;
    doublereal d = 0.0;
    doublereal l = 0.0;

    if (h > 1e-12) {
        d = delta / h;
        l = L_orig / h;
    }

    // 水平張力パラメータ x = H /(w*h) について
    doublereal x_param = 0.0;
    doublereal p0 = 0.0;
    doublereal H = 0.0; // 水平張力
    doublereal V = 0.0; // 鉛直張力

    if (d <= 0.0) {
        // たるんでいる場合
        H = 0.0;
        V = w_orig * h;
    } else if (h > 1e-12 && d < (std::sqrt(l*l - 1.0) - (l - 1.0))) {
        // カテナリー形状
        doublereal x1 = 0.0;
        doublereal x2 = 1.0e6;
        x_param = rtsafe_catenary_local(x1, x2, xacc_orig, d, l, p0);
        H = x_param * w_orig * h;
        V = w_orig * h * std::sqrt(1.0 + 2.0*x_param);
    }

    // ここから各内部ノードについて計算する
    std::vector<Vec3> node_positions(Seg_param); // 各内部ノードの初期位置
    node_positions[0] = fairlead_pos; // FP

    if (H > 1e-12) {
        // カテナリー形状に沿って配置
        doublereal a = H / w_orig; // カテナリーパラメータ

        // AP からの弧長を計算
        std::vector<doublereal> arc_length(Seg_param + 1);
        arc_length[0] = 0.0; // AP
        
        for (unsigned int i = 1; i <= Seg_param; ++i) {
            arc_length[i] = L_orig * static_cast<doublereal>(i) / static_cast<doublereal>(Seg_param);
        }

        // 弧長に対応する位置の計算
        for (unsigned int i = 1; i < Seg_param; ++i) {
            doublereal s = L_orig - arc_length[i]; // FP からの弧長

            // カテナリー曲線に沿った位置の計算
            doublereal x_local = 0.0;
            doublereal z_local = 0.0;

            if (p0 < 1e-12) {
                // 海底接触無しの場合
                doublereal beta = s / a;
                if (beta < 50.0) {
                    x_local = a*myasinh_local(std::sinh(L_APFP / a) - std::sinh(beta));
                    z_local = a*(std::cosh((L_APFP - x_local) / a) - std::cosh(L_APFP / a));
                } else {
                    x_local = L_APFP - s;
                    z_local = 0.0;
                }
            } else {
                // 海底接触ありの場合（簡略化）
                doublereal ratio = static_cast<doublereal>(i) / static_cast<doublereal>(Seg_param);
                x_local = L_APFP * (1.0 - ratio);
                z_local = h * (1.0 - ratio);
            }

            // グローバル座標系における各ノードの位置
            node_positions[i] = anchor_pos + horizontal_dir*x_local + Vec3(0.0, 0.0, z_local);
        }
    } else {
        // 垂直に垂れ下がる場合
        for (unsigned int i = 1; i < Seg_param; ++i) {
            double ratio = static_cast<doublereal>(i) / static_cast<doublereal>(Seg_param);
            doublereal z_offset = -h * ratio;
            node_positions[i] = fairlead_pos + Vec3(0.0, 0.0, z_offset);
        }
    }

    // 各ノードの初期位置と速度を設定
    for (unsigned int i = 1; i < Seg_param; ++i) {
        if (N_nodes_param[i] != 0) {

            // 位置の設定
            integer iFirstPosIndex = N_nodes_param[i] -> iGetFirstPositionIndex();
            X.PutCoef(iFirstPosIndex + 1, node_positions[i].dGet(1));  // X 座標
            X.PutCoef(iFirstPosIndex + 2, node_positions[i].dGet(2));  // Y 座標
            X.PutCoef(iFirstPosIndex + 3, node_positions[i].dGet(3));  // Z 座標

            // 速度は 0 で初期化
            integer iFirstVelIndex = N_nodes_param[i]->iGetFirstMomentumIndex();
            XP.PutCoef(iFirstVelIndex + 1, 0.0);  // Vx = 0
            XP.PutCoef(iFirstVelIndex + 2, 0.0);  // Vy = 0
            XP.PutCoef(iFirstVelIndex + 3, 0.0);  // Vz = 0
        }
    }
}

// Outputメソッド: シミュレーション結果の出力
void ModuleCatenaryLM::Output(OutputHandler& OH) const {
    if (!bToBeOutput()) return;
    if (!OH.UseText(OutputHandler::LOADABLE)) return;

    const integer lbl = GetLabel();

    // フェアリーダ座標
    if (N_nodes_param.empty() || N_nodes_param[0] == 0) {
        OH.Loadable() << lbl << "[Error] Fairlead node not available for output.\n";
        return;
    }
    const Vec3 posFL = N_nodes_param[0] -> GetXCurr();

    // フェアリーダに作用する軸方向力
    Vec3 Ffl(0.0, 0.0, 0.0);
    if (N_nodes_param.size() >= 2 && N_nodes_param[1] != 0) {
        Vec3 x0 = posFL;
        Vec3 x1 = N_nodes_param[1] -> GetXCurr();
        Vec3 v0 = N_nodes_param[0] -> GetVCurr();
        Vec3 v1 = N_nodes_param[1] -> GetVCurr();

        Vec3 dx = x1 - x0;
        doublereal l = dx.Norm();
        if (l > 1.e-12) {
            Vec3 t = dx / l;

            doublereal L0 = P_param[0].L0_seg;
            doublereal EA = P_param[0].EA_seg;
            doublereal CA = P_param[0].CA_seg;

            doublereal Fel = EA * (l - L0) / L0;
            Vec3 dv = v1 - v0;
            doublereal vrel = dv(1)*t(1) + dv(2)*t(2) + dv(3)*t(3);
            doublereal Fd = vrel + CA;
            doublereal Fax = Fel + Fd;
            if (Fax < 0.0) Fax = 0.0;

            Vec3 F = t * Fax;
            const doublereal fsf = FSF_orig.dGet();
            Ffl = F * fsf;
        }
    }
    OH.Loadable() << lbl << " FairleadPos "
                  << posFL.dGet(1) << " "
                  << posFL.dGet(2) << " "
                  << posFL.dGet(3) << " FairleadForce "
                  << Ffl.dGet(1)  << " "
                  << Ffl.dGet(2)  << " "
                  << Ffl.dGet(3)  << '\n';
}

// ================ ヤコビアン行列と残差ベクトルの次元を設定 ====================
// origin ではフェアリーダーポイントのみで 6 自由度で扱っていたが，ランプドマス法で内部ノードも全て管理する場合は大きく変わる
void ModuleCatenaryLM::WorkSpaceDim(integer* piNumRows, integer* piNumCols) const {
    unsigned int num_nodes = Seg_param;
    unsigned int dof_per_node = 6;

    *piNumRows = num_nodes * dof_per_node;
    *piNumCols = num_nodes * dof_per_node;
}

void ModuleCatenaryLM::InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const {
    unsigned int num_nodes = Seg_param;
    unsigned int dof_per_node = 6;

    *piNumRows = num_nodes * dof_per_node;
    *piNumCols = num_nodes * dof_per_node;
}

// ================ 残差 ==========================
SubVectorHandler& ModuleCatenaryLM::AssRes(
    SubVectorHandler& R,
    doublereal dCoef,
    const VectorHandler& XCurr,
    const VectorHandler& XPrimeCurr
) {

    const doublereal fsf = FSF_orig.dGet();

    const unsigned int ndof_tot = Seg_param * 6;
    R.ResizeReset(ndof_tot);

    for (unsigned int i = 0; i < Seg_param; ++i) {
        
        const unsigned int idx = i*6;

        // ====== 重力 + 慣性：ノード単位 ======
        const Vec3 Fg(0.0, 0.0, -node_mass_param[i]*g_gravity_param);
        Vec3 Fg_scaled = Fg*fsf;
        R.Add(idx+1, Fg_scaled);

        // 慣性 -M * a_lumped (a = dCoef*XPrime)
        const Vec3 a_like(XPrimeCurr(idx + 1), XPrimeCurr(idx + 2), XPrimeCurr(idx + 3));
        const Vec3 Fi = - a_like * dCoef * node_mass_param[i];
        R.Add(idx+1, Fi);

        // 海底接触があるならノード単位で反力を追加（押し返す）
        const doublereal z_i = N_nodes_param[i] -> GetXCurr().dGet(3);
        if (z_i < seabed_z_param) {
            const doublereal pen = seabed_z_param - z_i; // 海底侵入量（z 座標的な意味）
            const doublereal vz = N_nodes_param[i] -> GetVCurr().dGet(3); // 海底座標のクラス定義で下向き負，上向き正を決めている
            const doublereal Fz = Kseabed_param * pen - Cseabed_param * vz; // ばね + 減衰

            Vec3 Fseabed(0.0, 0.0, Fz*fsf);
            R.Add(idx+1, Fseabed); // +Z 方向の反力
        }

        // 回転 DOF (4 ,5, 6) はここでは 0 : ノード単位の計算の時は考慮しない
    }
    
    // ====== 軸方向の EA + CA：セグメント単位 ======

    for (unsigned int s = 0; s < Seg_param - 1; ++s) {
        const unsigned int i = s; // 左のノードのインデックス
        const unsigned int j = s + 1; // 右のノードのインデックス

        // 現在の位置と速度
        const Vec3 xi = N_nodes_param[i]->GetXCurr();
        const Vec3 xj = N_nodes_param[j]->GetXCurr();
        const Vec3 vi = N_nodes_param[i]->GetVCurr();
        const Vec3 vj = N_nodes_param[j]->GetVCurr();

        const Vec3 dx = xj - xi;
        const doublereal l = dx.Norm();
        if (l < 1.e-12) continue;

        const Vec3 t = dx / l;

        // 物性
        const doublereal L0 = P_param[s].L0_seg;
        const doublereal EA = P_param[s].EA_seg;
        const doublereal CA = P_param[s].CA_seg;

        // 弾性力と減衰力
        const doublereal Fel = EA * (l - L0) / L0;
        const Vec3 dv = vj - vi;
        const doublereal vrel = dv(1)*t(1) + dv(2)*t(2) + dv(3)*t(3);
        const doublereal Fd = vrel * CA;

        doublereal Faxial = Fel + Fd; // 弾性 + 減衰合力
        if (Faxial < 0.0) Faxial = 0.0; // 圧縮不可
        /* 
        // API の確認が取れたら実装する？　符号変化時の再線形化
        bool slack_now = (Fax <= 0.0);
        if (slack_now != segSlack_prev[s]) { // 状況が変わった場合
            pHints->bSetNotLinear(); // 再線形化を要求
            segSlack_prev[s] = slack_now;
        }
        if (slack_now) continue;
        */
        const Vec3 F = t * (Faxial * fsf); // 軸力ベクトル

        if (Faxial > 0.0) { // Faxial == 0 のときは何もしない
            const unsigned int idx_i = i*6;
            const unsigned int idx_j = j*6;

            R.Add(idx_i+1, -F); // 左ノード
            R.Add(idx_j+1, F); // 右ノード
        }
    }

    return R;
}

SubVectorHandler& ModuleCatenaryLM::InitialAssRes(
    SubVectorHandler& WorkVec,
    const VectorHandler& XCurr
) {
    return WorkVec;
}

// ============== 全体ヤコビアン ====================
VariableSubMatrixHandler& ModuleCatenaryLM::AssJac(
    VariableSubMatrixHandler& WH,
    doublereal dCoef,
    const VectorHandler& XCurr, 
    const VectorHandler& XPrimeCurr
) {
    const integer ndof_tot = Seg_param * 6; // ベクトルの大きさを決める int 変数を宣言して初期化
    FullSubMatrixHandler& K = WH.SetFull();
    K.ResizeReset(ndof_tot, ndof_tot); // 初期化（0 が入っている）

    // ============================
    // 海底ばね・ダンパ：各ノードの z 行に対角係数を追加：ノードループ
    // ============================

    for (unsigned int i = 0; i < Seg_param; ++i) {
        const unsigned int idx = i*6;
        const doublereal z_i = N_nodes_param[i] -> GetXCurr().dGet(3);

        if (z_i < seabed_z_param) { // z 座標によって付加
            const doublereal k_sea = Kseabed_param;
            const doublereal c_sea = - Cseabed_param * dCoef;

            K.PutCoef(idx+3, idx+3, k_sea + c_sea);
        }
    }

    // 回転 DOF を数値的に固定：大きな対角剛性をもたせる
    for (unsigned int i = 0; i < Seg_param; ++i) {
        const unsigned int idx = i * 6;
        for (int r = 4; r <= 6; ++r) { 
            K.PutCoef(idx + r, idx + r, Krot_lock_param);
        }
    }

    // =============================
    // EA / CA + 幾何剛性：セグメントループ
    // =============================
    for (unsigned int s = 0; s < Seg_param - 1; ++s) {

        // 隣接ノード index
        const unsigned int i = s;
        const unsigned int j = s + 1;
        const unsigned int row_i = i*6;
        const unsigned int row_j = j*6;

        // ======= 位置・速度・方向 ========
        Vec3 xi = N_nodes_param[i] -> GetXCurr();
        Vec3 xj = N_nodes_param[j] -> GetXCurr();
        Vec3 vi = N_nodes_param[i] -> GetVCurr();
        Vec3 vj = N_nodes_param[j] -> GetVCurr();

        Vec3 dx = xj - xi;
        doublereal l = dx.Norm();
        if (l < 1.e-12) continue;
        Vec3 t = dx / l; // 単位ベクトル

        // ====== 物性と係数 =======
        doublereal EA = P_param[s].EA_seg;
        doublereal CA = P_param[s].CA_seg;
        doublereal L0 = P_param[s].L0_seg;

        const doublereal k_ax = EA / L0;
        const doublereal c_ax = - CA * dCoef; // ここで符号を揃えた

        // ====== 軸力 Fel ========
        doublereal Fel = EA * (l - L0) / L0;
        if (Fel <= 0.0) continue; // 張力 0 と圧縮は剛性無しと判断

        // 基本ブロック Kt = (k_ax + c_ax)・(t × t)
        doublereal Klocal[3][3];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                
                // t⊗t 項：軸方向剛性 + 減衰 
                doublereal K_tt = (k_ax + c_ax) * t(r+1) * t(c+1);

                // 幾何剛性 kg·(I − t⊗t)
                doublereal delta = (r == c) ? 1.0 : 0.0;
                doublereal Kg = (Fel / l) * (delta - t(r+1) * t(c+1));
                
                Klocal[r][c] = K_tt + Kg;
            }
        }

        // 全体ヤコビアン行列の対応する 3x3 並進ブロックに K_local_3x3 を組み込む
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {

                const doublereal val = Klocal[r][c];

                // 左節点 i
                K.PutCoef(row_i + 1 + r, row_i + 1 + c, val);
                K.PutCoef(row_i + 1 + r, row_j + 1 + c, - val);

                // 右節点 j : 対称性で符号反転
                K.PutCoef(row_j + 1 + r, row_i + 1 + c, - val);
                K.PutCoef(row_j + 1 + r, row_j + 1 + c, val);
            }
        }
    }

    return WH;
}

VariableSubMatrixHandler& ModuleCatenaryLM::InitialAssJac(
    VariableSubMatrixHandler& WorkMat,
    const VectorHandler& /*X*/
) {
    WorkMat.SetNullMatrix();
    return WorkMat;
}

// この要素がMBDynのモデル内で接続している，あるいは管理しているノードの数を返す
// MBDyn は要素とノード間の接続情報を構築・管理するためにこの情報を利用する
unsigned int ModuleCatenaryLM::iGetNumConnectedNodes(void) const {
    return Seg_param; // これで完成のはず
}

// 後回し
std::ostream& ModuleCatenaryLM::Restart(std::ostream& out) const {
    out << "# ModuleCatenaryLM (Label: " << GetLabel() << ") Restart: Not implemented yet." << std::endl;
    return out;
}

// 初期化フェーズで特別な解析をするなら，初期のみ自由度を付与する
unsigned int ModuleCatenaryLM::iGetInitialNumDof(void) const {
    return Seg_param * 6;
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

// ==== ModuleCatenaryLM 要素が管理している i 番目のノードへのポインタを返すためのインターフェース ====
// MBDyn のソルバーや他のモジュールが，この要素に関連付けられたノードの情報（位置・速度・力など）にアクセスしたり，ノードの状態を変更したりするために使用する
// const 版：この関数自身が const であり，この関数を呼び出しても ModuleCatenaryLM オブジェクトの状態は変更されない
const Node* ModuleCatenaryLM::pGetNode(unsigned int i) const {
    if (i < N_nodes_param.size() && N_nodes_param[i] != 0) {
        return N_nodes_param[i];
    }
    return 0;
}

// 非 const 版：ノードの状態を変更することが可能
Node* ModuleCatenaryLM::pGetNode(unsigned int i) {
    if (i < N_nodes_param.size() && N_nodes_param[i] != 0) {
        return N_nodes_param[i];
    }
    return 0;
}

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
