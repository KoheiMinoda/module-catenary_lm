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

// ========= 流体力パラメータ構造体 =========
struct FluidForceParams {
    doublereal rho_water;
    doublereal cd_normal;
    doublereal cd_axial;
    doublereal ca_normal_x;
    doublereal ca_normal_y;
    doublereal ca_axial_z;
    doublereal cm_normal_x;
    doublereal cm_normal_y;
    doublereal cm_axial_z;
    doublereal diam_drag_normal;
    doublereal diam_drag_axial;
    doublereal kinematic_viscosity;
    
    FluidForceParams() :
        rho_water(1025.0),
        cd_normal(1.2), cd_axial(0.008),
        ca_normal_x(1.0), ca_normal_y(1.0), ca_axial_z(0.5),
        cm_normal_x(2.0), cm_normal_y(2.0), cm_axial_z(1.5),
        diam_drag_normal(0.05), diam_drag_axial(0.01592),
        kinematic_viscosity(1.35e-6)
    {}
};

// ========= 波浪パラメータ構造体 =========
struct WaveParams {
    doublereal wave_height;
    doublereal wave_period;
    doublereal wave_direction;
    doublereal water_depth;
    doublereal wave_length;
    
    WaveParams() :
        wave_height(0.0), wave_period(20.0), wave_direction(0.0),
        water_depth(320.0), wave_length(0.0)
    {
        // 波長計算（深水波近似）
        wave_length = (9.80665 * wave_period * wave_period) / (2.0 * M_PI);
    }
};

// ========= 海流パラメータ構造体 =========
struct CurrentParams {
    doublereal surface_velocity;
    doublereal bottom_velocity;
    doublereal direction;
    
    CurrentParams() :
        surface_velocity(0.0), bottom_velocity(0.0), direction(0.0)
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
        unsigned int iGetPrivDataIdx(const char *s) const;
        doublereal dGetPrivData(unsigned int i) const;
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
        doublereal contact_diameter;            // 接触径
        doublereal mu_lateral;                  // 横方向摩擦係数
        doublereal smooth_clearance;            // スムージングクリアランス
        doublereal smooth_eps;                  // スムージングイプシロン
        
        // 材料特性
        doublereal poisson_ratio;               // ポアソン比
        doublereal struct_damp_ratio;           // 構造減衰比
        doublereal p_atmospheric;               // 大気圧
        
        // レイリー減衰
        doublereal rayleigh_alpha;              // 質量比例減衰
        doublereal rayleigh_beta;               // 剛性比例減衰
        
        // 時間関連
        doublereal simulation_time;             // シミュレーション時間
        doublereal prev_time;                   // 前のタイムステップの時間
        doublereal ramp_time;                   // ランプアップ時間
        
        // 流体力パラメータ
        FluidForceParams fluid_params;
        WaveParams wave_params;
        CurrentParams current_params;
        
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
        
        // 流体力計算関数（Pythonコードから移植）
        Vec3 GetCurrentVelocity(doublereal z_coord, doublereal t) const;
        void GetWaveVelocityAcceleration(doublereal x, doublereal z, doublereal t, 
                                       Vec3& velocity, Vec3& acceleration) const;
        Vec3 ComputeFroudeKrylovForce(const Vec3& seg_vector, doublereal diameter, 
                                    const Vec3& fluid_acc) const;
        Vec3 ComputeAddedMassForce(const Vec3& seg_vector, doublereal seg_length, 
                                 doublereal diameter, const Vec3& structure_acc, 
                                 const Vec3& fluid_acc) const;
        Vec3 ComputeDragForce(const Vec3& seg_vector, doublereal seg_length, 
                            doublereal diam_normal, doublereal diam_axial,
                            const Vec3& rel_vel, doublereal cd_normal, 
                            doublereal cd_axial) const;
        
        // 力計算関数
        Vec3 ComputeAxialForce(const Vec3& pos1, const Vec3& pos2, const Vec3& vel1, 
                             const Vec3& vel2, doublereal L0) const;
        Vec3 ComputeSeabedForce(const VirtualNode& node) const;
        Vec3 ComputeRayleighDamping(const VirtualNode& node, 
                                  const std::vector<VirtualNode>& neighbors) const;
        
        // ユーティリティ関数
        doublereal CalculateExternalPressure(doublereal z_coord) const;
        doublereal SolveDispersion(doublereal omega, doublereal depth) const;
        void GetLocalCoordinateSystem(const Vec3& seg_vector, Vec3& t_vec, 
                                    Vec3& n1_vec, Vec3& n2_vec) const;
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
        contact_diameter(0.18), mu_lateral(0.0),
        smooth_clearance(0.01), smooth_eps(0.01),
        poisson_ratio(0.0), struct_damp_ratio(0.0), p_atmospheric(101325.0),
        rayleigh_alpha(0.0), rayleigh_beta(0.0),
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
            "              [ contact_diameter, diameter, ]\n"
            "              [ friction, mu_lateral, ]\n"
            "              [ material, poisson_ratio, struct_damping, ]\n"
            "              [ rayleigh, alpha, beta, ]\n"
            "              [ fluid_params, rho_water, cd_normal, cd_axial, ]\n"
            "              [ added_mass, ca_normal_x, ca_normal_y, ca_axial_z, ]\n"
            "              [ inertia_coeff, cm_normal_x, cm_normal_y, cm_axial_z, ]\n"
            "              [ drag_diameters, diam_normal, diam_axial, ]\n"
            "              [ wave, height, period, direction, water_depth, ]\n"
            "              [ current, surface_vel, bottom_vel, direction, ]\n"
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

    // 必須パラメータの読み込み
    g_pNode = dynamic_cast<const StructNode *>(pDM->ReadNode(HP, Node::STRUCTURAL));
    L = HP.GetReal();
    w = HP.GetReal();
    xacc = HP.GetReal();
    APx = HP.GetReal();
    APy = HP.GetReal();
    APz = HP.GetReal();

    // オプショナルパラメータの読み込み
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
    if (HP.IsKeyWord("contact_diameter")) {
        contact_diameter = HP.GetReal();
    }
    if (HP.IsKeyWord("friction")) {
        mu_lateral = HP.GetReal();
    }
    if (HP.IsKeyWord("material")) {
        poisson_ratio = HP.GetReal();
        struct_damp_ratio = HP.GetReal();
    }
    if (HP.IsKeyWord("rayleigh")) {
        rayleigh_alpha = HP.GetReal();
        rayleigh_beta = HP.GetReal();
    }
    
    // 流体力パラメータ
    if (HP.IsKeyWord("fluid_params")) {
        fluid_params.rho_water = HP.GetReal();
        fluid_params.cd_normal = HP.GetReal();
        fluid_params.cd_axial = HP.GetReal();
    }
    if (HP.IsKeyWord("added_mass")) {
        fluid_params.ca_normal_x = HP.GetReal();
        fluid_params.ca_normal_y = HP.GetReal();
        fluid_params.ca_axial_z = HP.GetReal();
    }
    if (HP.IsKeyWord("inertia_coeff")) {
        fluid_params.cm_normal_x = HP.GetReal();
        fluid_params.cm_normal_y = HP.GetReal();
        fluid_params.cm_axial_z = HP.GetReal();
    }
    if (HP.IsKeyWord("drag_diameters")) {
        fluid_params.diam_drag_normal = HP.GetReal();
        fluid_params.diam_drag_axial = HP.GetReal();
    }
    
    // 波浪パラメータ
    if (HP.IsKeyWord("wave")) {
        wave_params.wave_height = HP.GetReal();
        wave_params.wave_period = HP.GetReal();
        wave_params.wave_direction = HP.GetReal();
        wave_params.water_depth = HP.GetReal();
        // 波長再計算
        wave_params.wave_length = (g_gravity * wave_params.wave_period * wave_params.wave_period) / (2.0 * M_PI);
    }
    
    // 海流パラメータ
    if (HP.IsKeyWord("current")) {
        current_params.surface_velocity = HP.GetReal();
        current_params.bottom_velocity = HP.GetReal();
        current_params.direction = HP.GetReal();
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
        << "FluidForces: enabled "
        << "Wave: H=" << wave_params.wave_height << "m T=" << wave_params.wave_period << "s "
        << std::endl;
}

ModuleCatenaryLM::~ModuleCatenaryLM(void)
{
    // destroy private data
    NO_OP;
}

doublereal ModuleCatenaryLM::GetRampFactor(doublereal current_time) const {
    if (current_time <= 0.0) {
        return 0.0;
    } else if (current_time >= ramp_time) {
        return 1.0;
    } else {
        return current_time / ramp_time;
    }
}

// ========= カテナリー理論関数（元コードから継承） =========
double ModuleCatenaryLM::myasinh(double X) 
{
    return std::log(X + std::sqrt(X * X + 1));
}

double ModuleCatenaryLM::myacosh(double X) 
{
    if (X < 1.0) {
        if (X > 1.0 - 1e-9) {
            X = 1.0;
        } else {
            throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
        }
    }
    return std::log(X + std::sqrt(X * X - 1));
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

// ========= 流体力計算関数（Pythonコードから移植） =========

// 海流速度計算
Vec3 ModuleCatenaryLM::GetCurrentVelocity(doublereal z_coord, doublereal t) const
{
    doublereal depth_ratio = std::abs(z_coord) / wave_params.water_depth;
    depth_ratio = std::min(1.0, std::max(0.0, depth_ratio));
    
    doublereal current_magnitude = current_params.surface_velocity * (1.0 - depth_ratio) + 
                                 current_params.bottom_velocity * depth_ratio;
    
    doublereal current_dir_rad = current_params.direction * M_PI / 180.0;
    doublereal u_current = current_magnitude * std::cos(current_dir_rad);
    doublereal v_current = current_magnitude * std::sin(current_dir_rad);
    
    return Vec3(u_current, v_current, 0.0);
}

// 分散関係式求解
doublereal ModuleCatenaryLM::SolveDispersion(doublereal omega, doublereal depth) const
{
    doublereal k = omega * omega / g_gravity;
    const doublereal tol = 1e-6;
    const int maxit = 50;
    
    for (int i = 0; i < maxit; ++i) {
        doublereal tanh_kh = std::tanh(k * depth);
        doublereal f = g_gravity * k * tanh_kh - omega * omega;
        doublereal df = g_gravity * tanh_kh + g_gravity * k * depth * (1.0 - tanh_kh * tanh_kh);
        doublereal dk = -f / df;
        k += dk;
        if (std::abs(dk) < tol) {
            break;
        }
    }
    return k;
}

// 波浪速度・加速度計算
void ModuleCatenaryLM::GetWaveVelocityAcceleration(doublereal x, doublereal z, doublereal t, 
                                                 Vec3& velocity, Vec3& acceleration) const
{
    if (wave_params.wave_height < 1e-6) {
        velocity = Vec3(0.0, 0.0, 0.0);
        acceleration = Vec3(0.0, 0.0, 0.0);
        return;
    }
    
    doublereal k = 2.0 * M_PI / wave_params.wave_length;
    doublereal omega = 2.0 * M_PI / wave_params.wave_period;
    doublereal amplitude = wave_params.wave_height / 2.0;
    
    doublereal wave_dir_rad = wave_params.wave_direction * M_PI / 180.0;
    doublereal kx = k * std::cos(wave_dir_rad);
    doublereal ky = k * std::sin(wave_dir_rad);
    
    doublereal phase = kx * x + ky * 0.0 - omega * t;
    
    doublereal depth_from_surface = std::abs(z);
    doublereal cosh_factor, sinh_factor;
    
    if (depth_from_surface >= wave_params.water_depth) {
        cosh_factor = std::cosh(k * wave_params.water_depth) / std::sinh(k * wave_params.water_depth);
        sinh_factor = 1.0 / std::sinh(k * wave_params.water_depth);
    } else {
        cosh_factor = std::cosh(k * (wave_params.water_depth - depth_from_surface)) / 
                     std::sinh(k * wave_params.water_depth);
        sinh_factor = std::sinh(k * (wave_params.water_depth - depth_from_surface)) / 
                     std::sinh(k * wave_params.water_depth);
    }
    
    doublereal u_wave = amplitude * omega * cosh_factor * std::cos(phase) * std::cos(wave_dir_rad);
    doublereal v_wave = amplitude * omega * cosh_factor * std::cos(phase) * std::sin(wave_dir_rad);
    doublereal w_wave = amplitude * omega * sinh_factor * std::sin(phase);
    
    doublereal du_dt = -amplitude * omega * omega * cosh_factor * std::sin(phase) * std::cos(wave_dir_rad);
    doublereal dv_dt = -amplitude * omega * omega * cosh_factor * std::sin(phase) * std::sin(wave_dir_rad);
    doublereal dw_dt = amplitude * omega * omega * sinh_factor * std::cos(phase);
    
    velocity = Vec3(u_wave, v_wave, w_wave);
    acceleration = Vec3(du_dt, dv_dt, dw_dt);
}

// 局所座標系計算
void ModuleCatenaryLM::GetLocalCoordinateSystem(const Vec3& seg_vector, Vec3& t_vec, 
                                               Vec3& n1_vec, Vec3& n2_vec) const
{
    doublereal seg_length = seg_vector.Norm();
    if (seg_length < 1e-9) {
        t_vec = Vec3(1.0, 0.0, 0.0);
        n1_vec = Vec3(0.0, 1.0, 0.0);
        n2_vec = Vec3(0.0, 0.0, 1.0);
        return;
    }
    
    t_vec = seg_vector / seg_length;
    
    if (std::abs(t_vec.dGet(3)) > 0.99) {
        n1_vec = Vec3(1.0, 0.0, 0.0);
        n2_vec = Vec3(0.0, 1.0, 0.0);
    } else {
        Vec3 temp_z(0.0, 0.0, 1.0);
        n1_vec = t_vec.Cross(temp_z);
        n1_vec = n1_vec / n1_vec.Norm();
        n2_vec = t_vec.Cross(n1_vec);
        n2_vec = n2_vec / n2_vec.Norm();
    }
}

// Froude-Krylov力計算
Vec3 ModuleCatenaryLM::ComputeFroudeKrylovForce(const Vec3& seg_vector, doublereal diameter, 
                                              const Vec3& fluid_acc) const
{
    doublereal seg_length = seg_vector.Norm();
    if (seg_length < 1e-12) {
        return Vec3(0.0, 0.0, 0.0);
    }
    
    doublereal volume = M_PI * (diameter/2.0) * (diameter/2.0) * seg_length;
    
    Vec3 t_vec, n1_vec, n2_vec;
    GetLocalCoordinateSystem(seg_vector, t_vec, n1_vec, n2_vec);
    
    doublereal fluid_acc_axial_scalar = fluid_acc.Dot(t_vec);
    Vec3 fluid_acc_axial = t_vec * fluid_acc_axial_scalar;
    Vec3 fluid_acc_normal = fluid_acc - fluid_acc_axial;
    
    doublereal fluid_acc_normal_x = fluid_acc_normal.Dot(n1_vec);
    doublereal fluid_acc_normal_y = fluid_acc_normal.Dot(n2_vec);
    
    Vec3 F_FK_x = n1_vec * (fluid_params.rho_water * volume * fluid_params.cm_normal_x * fluid_acc_normal_x);
    Vec3 F_FK_y = n2_vec * (fluid_params.rho_water * volume * fluid_params.cm_normal_y * fluid_acc_normal_y);
    Vec3 F_FK_z = t_vec * (fluid_params.rho_water * volume * fluid_params.cm_axial_z * fluid_acc_axial_scalar);
    
    return F_FK_x + F_FK_y + F_FK_z;
}

// 付加質量力計算
Vec3 ModuleCatenaryLM::ComputeAddedMassForce(const Vec3& seg_vector, doublereal seg_length, 
                                           doublereal diameter, const Vec3& structure_acc, 
                                           const Vec3& fluid_acc) const
{
    doublereal volume = M_PI * (diameter/2.0) * (diameter/2.0) * seg_length;
    
    doublereal seg_length_calc = seg_vector.Norm();
    if (seg_length_calc < 1e-9) {
        return Vec3(0.0, 0.0, 0.0);
    }
    
    Vec3 t_vec, n1_vec, n2_vec;
    GetLocalCoordinateSystem(seg_vector, t_vec, n1_vec, n2_vec);
    
    Vec3 acc_rel = structure_acc - fluid_acc;
    
    doublereal acc_rel_axial_scalar = acc_rel.Dot(t_vec);
    Vec3 acc_rel_axial = t_vec * acc_rel_axial_scalar;
    Vec3 acc_rel_normal = acc_rel - acc_rel_axial;
    
    doublereal acc_normal_x = acc_rel_normal.Dot(n1_vec);
    doublereal acc_normal_y = acc_rel_normal.Dot(n2_vec);
    
    Vec3 F_AM_x = n1_vec * (-fluid_params.rho_water * volume * fluid_params.ca_normal_x * acc_normal_x);
    Vec3 F_AM_y = n2_vec * (-fluid_params.rho_water * volume * fluid_params.ca_normal_y * acc_normal_y);
    Vec3 F_AM_z = t_vec * (-fluid_params.rho_water * volume * fluid_params.ca_axial_z * acc_rel_axial_scalar);
    
    return F_AM_x + F_AM_y + F_AM_z;
}

// 抗力計算
Vec3 ModuleCatenaryLM::ComputeDragForce(const Vec3& seg_vector, doublereal seg_length, 
                                      doublereal diam_normal, doublereal diam_axial,
                                      const Vec3& rel_vel, doublereal cd_normal, 
                                      doublereal cd_axial) const
{
    doublereal seg_length_calc = seg_vector.Norm();
    if (seg_length_calc < 1e-9) {
        return Vec3(0.0, 0.0, 0.0);
    }
    
    Vec3 t_vec, n1_vec, n2_vec;
    GetLocalCoordinateSystem(seg_vector, t_vec, n1_vec, n2_vec);
    
    doublereal vel_t_scalar = rel_vel.Dot(t_vec);
    Vec3 vel_t = t_vec * vel_t_scalar;
    Vec3 vel_n = rel_vel - vel_t;
    
    doublereal mag_t = std::abs(vel_t_scalar);
    doublereal mag_n = vel_n.Norm();
    
    doublereal area_n = diam_normal * seg_length;
    doublereal area_t = M_PI * diam_axial * seg_length;
    
    Vec3 Fd_n(0.0, 0.0, 0.0);
    if (mag_n > 1e-6) {
        Fd_n = vel_n * (0.5 * fluid_params.rho_water * cd_normal * area_n * mag_n);
    }
    
    Vec3 Fd_t(0.0, 0.0, 0.0);
    if (mag_t > 1e-6) {
        Fd_t = vel_t * (0.5 * fluid_params.rho_water * cd_axial * area_t * mag_t);
    }
    
    return Fd_n + Fd_t;
}

// ========= 力計算関数 =========

// 外圧計算
doublereal ModuleCatenaryLM::CalculateExternalPressure(doublereal z_coord) const
{
    doublereal depth = std::abs(std::min(0.0, z_coord));
    return p_atmospheric + fluid_params.rho_water * g_gravity * depth;
}

// 軸力計算（Pythonコードから移植・改良）
Vec3 ModuleCatenaryLM::ComputeAxialForce(const Vec3& pos1, const Vec3& pos2, const Vec3& vel1, 
                                       const Vec3& vel2, doublereal L0) const
{
    Vec3 dx = pos2 - pos1;
    doublereal l_current = dx.Norm();
    
    if (l_current < 1e-12) {
        return Vec3(0.0, 0.0, 0.0);
    }
    
    Vec3 t_vec = dx / l_current;
    
    // 中点での外圧計算
    doublereal mid_z = 0.5 * (pos1.dGet(3) + pos2.dGet(3));
    doublereal Po = CalculateExternalPressure(mid_z);
    doublereal Ao = M_PI * line_diameter * line_diameter;
    
    // ポアソン効果
    doublereal poisson_effect = -2.0 * poisson_ratio * Po * Ao;
    doublereal pressure_term = Po * Ao;
    
    // ひずみ計算
    doublereal strain = (l_current - L0) / L0;
    
    // 相対速度と軸方向速度変化率
    Vec3 rel_vel = vel2 - vel1;
    doublereal dl_dt = dx.Dot(rel_vel) / l_current;
    doublereal strain_rate = dl_dt / L0;
    
    // 弾性力（引張のみ）
    doublereal F_elastic = 0.0;
    if (strain > 0.0) {
        F_elastic = EA * strain + pressure_term;
    }
    
    // 構造減衰による力
    doublereal F_strain_rate_damping = EA * struct_damp_ratio * strain_rate;
    
    // 軸減衰による力
    doublereal F_axial_damping = CA * dl_dt / L0;
    
    // 総軸力
    doublereal total_tension = F_elastic + F_strain_rate_damping + F_axial_damping + poisson_effect;
    
    // 力の制限（負の値を防ぐ）
    total_tension = std::max(0.0, std::min(total_tension, 100.0e9));
    
    return t_vec * total_tension;
}

// 海底接触力計算（改良版）
Vec3 ModuleCatenaryLM::ComputeSeabedForce(const VirtualNode& node) const
{
    Vec3 F_seabed(0.0, 0.0, 0.0);
    
    doublereal z = node.position.dGet(3);
    doublereal raw_penetration = seabed_z - z;
    doublereal p_eff = raw_penetration + smooth_clearance + contact_diameter;
    
    if (p_eff <= 0.0) {
        return F_seabed;
    }
    
    // スムージング付き法線力
    doublereal radius = contact_diameter / 2.0;
    doublereal K_eff = K_seabed * radius * p_eff;
    doublereal F_normal = K_eff * (std::sqrt(p_eff*p_eff + smooth_eps*smooth_eps) - smooth_eps);
    
    if (F_normal < 0.0) {
        F_normal = 0.0;
    }
    
    // 法線方向力
    Vec3 F_norm_vec(0.0, 0.0, F_normal);
    
    // 摩擦力計算
    Vec3 vel_lateral(node.velocity.dGet(1), node.velocity.dGet(2), 0.0);
    doublereal mag_lateral = vel_lateral.Norm();
    
    Vec3 F_fric_vec(0.0, 0.0, 0.0);
    const doublereal v_slip_tol = 1e-5;
    if (mag_lateral > v_slip_tol) {
        F_fric_vec = vel_lateral * (-mu_lateral * F_normal / mag_lateral);
    }
    
    // 減衰力
    doublereal F_damping_z = -C_seabed * node.velocity.dGet(3);
    if (F_damping_z < 0.0) F_damping_z = 0.0; // 海底から離れる方向の減衰は無し
    
    F_seabed = F_norm_vec + F_fric_vec + Vec3(0.0, 0.0, F_damping_z);
    
    return F_seabed;
}

// レイリー減衰力計算
Vec3 ModuleCatenaryLM::ComputeRayleighDamping(const VirtualNode& node, 
                                            const std::vector<VirtualNode>& neighbors) const
{
    Vec3 F_damp(0.0, 0.0, 0.0);
    
    // 質量比例減衰
    F_damp += node.velocity * (-rayleigh_alpha * node.mass);
    
    // 剛性比例減衰（隣接ノードとの相対速度に基づく）
    for (const auto& neighbor : neighbors) {
        Vec3 rel_pos = neighbor.position - node.position;
        Vec3 rel_vel = neighbor.velocity - node.velocity;
        
        doublereal seg_length = rel_pos.Norm();
        if (seg_length < 1e-9) continue;
        
        Vec3 t_vec = rel_pos / seg_length;
        doublereal rel_vel_axial = rel_vel.Dot(t_vec);
        
        doublereal K_el = EA / (L / static_cast<doublereal>(NUM_SEGMENTS));
        doublereal Fd_scalar = rayleigh_beta * K_el * rel_vel_axial;
        
        F_damp += t_vec * Fd_scalar;
    }
    
    return F_damp;
}

// ========= カテナリー理論による仮想ノード初期位置設定（改良版） =========
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
    doublereal h = std::abs(FP_AP.dGet(3));
    doublereal L_APFP = std::sqrt(FP_AP.dGet(1)*FP_AP.dGet(1) + FP_AP.dGet(2)*FP_AP.dGet(2));
    
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
                doublereal x1 = xacc;
                doublereal x2 = std::max(100.0, l*l*2.0);
                
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

// ========= 仮想ノード更新（改良版・流体力統合） =========
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
        
        // 1. 重力
        Vec3 F_gravity(0.0, 0.0, -virtual_nodes[i].mass * g_gravity * ramp_factor);
        F_total += F_gravity;
        
        // 2. 前のセグメントからの軸力
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
        
        // 3. 次のセグメントからの軸力
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
        
        // 4. 海底反力
        Vec3 F_seabed = ComputeSeabedForce(virtual_nodes[i]);
        F_total += F_seabed * ramp_factor;
        
        // 5. 流体力計算（Pythonコードの高精度版を統合）
        Vec3 pos_mid = virtual_nodes[i].position;
        Vec3 vel_mid = virtual_nodes[i].velocity;
        Vec3 acc_mid = virtual_nodes[i].acceleration;
        
        // セグメントベクトル計算（前後の平均）
        Vec3 seg_vector_prev, seg_vector_next;
        if (i == 0) {
            seg_vector_prev = virtual_nodes[i].position - fairlead_pos;
        } else {
            seg_vector_prev = virtual_nodes[i].position - virtual_nodes[i-1].position;
        }
        
        if (i == virtual_nodes.size() - 1) {
            seg_vector_next = anchor_pos - virtual_nodes[i].position;
        } else {
            seg_vector_next = virtual_nodes[i+1].position - virtual_nodes[i].position;
        }
        
        Vec3 seg_vector_avg = (seg_vector_prev + seg_vector_next) * 0.5;
        
        // 流体速度・加速度取得
        Vec3 u_current = GetCurrentVelocity(pos_mid.dGet(3), simulation_time);
        Vec3 u_wave, a_wave;
        GetWaveVelocityAcceleration(pos_mid.dGet(1), pos_mid.dGet(3), simulation_time, u_wave, a_wave);
        
        Vec3 u_fluid = u_current + u_wave;
        Vec3 a_fluid = a_wave;
        
        // 5a. Froude-Krylov力
        Vec3 F_FK = ComputeFroudeKrylovForce(seg_vector_avg, line_diameter, a_fluid);
        F_total += F_FK * ramp_factor;
        
        // 5b. 付加質量力
        Vec3 F_AM = ComputeAddedMassForce(seg_vector_avg, segment_length, line_diameter, 
                                        acc_mid, a_fluid);
        F_total += F_AM * ramp_factor;
        
        // 5c. 抗力
        Vec3 u_rel = u_fluid - vel_mid;
        Vec3 F_drag = ComputeDragForce(seg_vector_avg, segment_length, 
                                     fluid_params.diam_drag_normal, fluid_params.diam_drag_axial,
                                     u_rel, fluid_params.cd_normal, fluid_params.cd_axial);
        F_total += F_drag * ramp_factor;
        
        // 6. レイリー減衰力
        std::vector<VirtualNode> neighbors;
        if (i > 0) neighbors.push_back(virtual_nodes[i-1]);
        if (i < virtual_nodes.size() - 1) neighbors.push_back(virtual_nodes[i+1]);
        
        Vec3 F_rayleigh = ComputeRayleighDamping(virtual_nodes[i], neighbors);
        F_total += F_rayleigh;
        
        // 7. 簡易構造減衰（数値安定化）
        Vec3 F_structural_damping = virtual_nodes[i].velocity * (-virtual_nodes[i].mass * 0.05);
        F_total += F_structural_damping;
        
        // 8. 実効質量による加速度計算（付加質量効果を考慮）
        doublereal volume_node = M_PI * (line_diameter/2.0) * (line_diameter/2.0) * segment_length;
        doublereal added_mass_scalar = fluid_params.rho_water * volume_node * 
                                     (fluid_params.ca_normal_x + fluid_params.ca_normal_y + fluid_params.ca_axial_z) / 3.0;
        doublereal effective_mass = virtual_nodes[i].mass + added_mass_scalar;
        
        // 加速度計算
        if (effective_mass > 1e-12) {
            virtual_nodes[i].acceleration = F_total / effective_mass;
            
            // 加速度制限（数値安定化）
            doublereal acc_norm = virtual_nodes[i].acceleration.Norm();
            doublereal max_acc = 50.0 * g_gravity; // より現実的な制限値
            if (acc_norm > max_acc) {
                virtual_nodes[i].acceleration = virtual_nodes[i].acceleration * (max_acc / acc_norm);
            }
        }
        
        // 9. 速度・位置更新（改良Euler法）
        Vec3 vel_old = virtual_nodes[i].velocity;
        virtual_nodes[i].velocity += virtual_nodes[i].acceleration * dt;
        
        // 速度制限（数値安定化）
        doublereal vel_norm = virtual_nodes[i].velocity.Norm();
        doublereal max_vel = 100.0; // より現実的な制限値
        if (vel_norm > max_vel) {
            virtual_nodes[i].velocity = virtual_nodes[i].velocity * (max_vel / vel_norm);
        }
        
        // 位置更新（台形則による改良）
        Vec3 vel_avg = (vel_old + virtual_nodes[i].velocity) * 0.5;
        virtual_nodes[i].position += vel_avg * dt;
        
        // 10. 海底貫入防止
        if (virtual_nodes[i].position.dGet(3) < seabed_z) {
            virtual_nodes[i].position = Vec3(virtual_nodes[i].position.dGet(1),
                                           virtual_nodes[i].position.dGet(2),
                                           seabed_z + 0.001); // より小さな余裕
            if (virtual_nodes[i].velocity.dGet(3) < 0.0) {
                virtual_nodes[i].velocity = Vec3(virtual_nodes[i].velocity.dGet(1),
                                               virtual_nodes[i].velocity.dGet(2),
                                               0.0);
            }
        }
    }
}

// ========= ランプドマス法による総合力計算（流体力統合版） =========
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
    
    // 第1セグメント（フェアリーダー - 第1仮想ノード）の力
    
    // 1. 軸力
    Vec3 F_axial = ComputeAxialForce(fairlead_pos, virtual_nodes[0].position,
                                   fairlead_vel, virtual_nodes[0].velocity, segment_length);
    F_total += F_axial * ramp_factor;
    
    // 2. 第1セグメントの流体力（フェアリーダーに作用する分）
    Vec3 seg_vector = virtual_nodes[0].position - fairlead_pos;
    Vec3 pos_mid = (fairlead_pos + virtual_nodes[0].position) * 0.5;
    Vec3 vel_mid = (fairlead_vel + virtual_nodes[0].velocity) * 0.5;
    
    // フェアリーダーの加速度（MBDynから取得）
    Vec3 fairlead_acc = g_pNode->GetXPPCurr();
    Vec3 acc_mid = (fairlead_acc + virtual_nodes[0].acceleration) * 0.5;
    
    // 流体速度・加速度
    Vec3 u_current = GetCurrentVelocity(pos_mid.dGet(3), simulation_time);
    Vec3 u_wave, a_wave;
    GetWaveVelocityAcceleration(pos_mid.dGet(1), pos_mid.dGet(3), simulation_time, u_wave, a_wave);
    
    Vec3 u_fluid = u_current + u_wave;
    Vec3 a_fluid = a_wave;
    
    // 流体力計算
    Vec3 F_FK_seg1 = ComputeFroudeKrylovForce(seg_vector, line_diameter, a_fluid);
    Vec3 F_AM_seg1 = ComputeAddedMassForce(seg_vector, segment_length, line_diameter, 
                                         acc_mid, a_fluid);
    Vec3 u_rel_seg1 = u_fluid - vel_mid;
    Vec3 F_drag_seg1 = ComputeDragForce(seg_vector, segment_length, 
                                      fluid_params.diam_drag_normal, fluid_params.diam_drag_axial,
                                      u_rel_seg1, fluid_params.cd_normal, fluid_params.cd_axial);
    
    // フェアリーダーに作用する流体力（第1セグメントの半分）
    Vec3 F_fluid_fairlead = (F_FK_seg1 + F_AM_seg1 + F_drag_seg1) * 0.5 * ramp_factor;
    F_total += F_fluid_fairlead;
    
    // 3. 集中質量による慣性力補正
    // 第1仮想ノードの加速度がフェアリーダーに与える反力
    Vec3 pos_rel = virtual_nodes[0].position - fairlead_pos;
    if (pos_rel.Norm() > 1e-6) {
        Vec3 F_inertial = virtual_nodes[0].acceleration * (-virtual_nodes[0].mass * 0.1); // 10%の結合効果
        F_total += F_inertial * ramp_factor;
    }
    
    // モーメントは考慮しない（ライン要素のため）
    M_total = Vec3(0.0, 0.0, 0.0);
    
    // デバッグ出力（必要に応じて）
    if (simulation_time - prev_time > 0.1) { // 0.1秒ごと
        static doublereal last_output_time = -1.0;
        if (simulation_time - last_output_time > 1.0) { // 1秒ごと
            std::cout << "Time: " << simulation_time 
                      << " F_total: (" << F_total.dGet(1) << ", " << F_total.dGet(2) << ", " << F_total.dGet(3) << ")"
                      << " F_axial_norm: " << F_axial.Norm()
                      << " F_fluid_norm: " << F_fluid_fairlead.Norm()
                      << " Ramp: " << ramp_factor
                      << std::endl;
            last_output_time = simulation_time;
        }
    }
}

// ========= MBDyn インターフェース =========
void ModuleCatenaryLM::Output(OutputHandler& OH) const
{
    if (bToBeOutput()) {
        if (OH.UseText(OutputHandler::LOADABLE)) {
            const Vec3& FP = g_pNode->GetXCurr();
            const Vec3& FV = g_pNode->GetVCurr();
            
            // 基本情報出力
            OH.Loadable() << GetLabel()
                << " " << FP.dGet(1)      // フェアリーダー位置
                << " " << FP.dGet(2)
                << " " << FP.dGet(3)
                << " " << FV.dGet(1)      // フェアリーダー速度
                << " " << FV.dGet(2)
                << " " << FV.dGet(3)
                << " " << virtual_nodes.size()  // 仮想ノード数
                << " " << simulation_time       // シミュレーション時間
                << " " << GetRampFactor(simulation_time)  // ランプファクター
                << std::endl;
                
            // 詳細情報（オプション）
            if (virtual_nodes.size() > 0) {
                // 第1ノードと最終ノードの情報
                const VirtualNode& first_node = virtual_nodes[0];
                const VirtualNode& last_node = virtual_nodes.back();
                
                OH.Loadable() << GetLabel() << "_detail"
                    << " " << first_node.position.dGet(1) << " " << first_node.position.dGet(2) << " " << first_node.position.dGet(3)
                    << " " << last_node.position.dGet(1) << " " << last_node.position.dGet(2) << " " << last_node.position.dGet(3)
                    << " " << first_node.velocity.Norm() << " " << last_node.velocity.Norm()
                    << " " << first_node.acceleration.Norm() << " " << last_node.acceleration.Norm()
                    << std::endl;
            }
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
    // ランプドマス法では線形化ヤコビアンを省略（陽解法のため）
    // 必要に応じて線形化項を追加可能
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

    // 時間更新判定（MBDynの標準的な方法）
    // dCoefは通常、時間積分係数だが、時間情報取得のため使用
    doublereal current_time = simulation_time + 0.001; // 仮の時間ステップ増分
    
    // 実際のMBDyn環境では以下のような方法で時間取得
    // const DataManager* pDM = dynamic_cast<const DataManager*>(GetDataManager());
    // if (pDM) current_time = pDM->dGetTime();
    
    if (current_time > prev_time + 1e-12) {
        doublereal dt = current_time - prev_time;
        if (dt < 1e-12) dt = 0.001; // デフォルトタイムステップ
        simulation_time = current_time;
        
        // 仮想ノード更新（妥当なタイムステップのみ）
        if (dt > 1e-12 && dt < 0.1) {
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
    return 12; // 位置3 + 速度3 + 加速度3 + その他3
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
    return out << "# ModuleCatenaryLM: restart not implemented" << std::endl;
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

// ========= プライベートデータアクセス（オプション） =========
unsigned int ModuleCatenaryLM::iGetPrivDataIdx(const char *s) const
{
    // プライベートデータへのアクセス名定義
    if (strcmp(s, "x1") == 0) return 1;
    if (strcmp(s, "y1") == 0) return 2;
    if (strcmp(s, "z1") == 0) return 3;
    if (strcmp(s, "vx1") == 0) return 4;
    if (strcmp(s, "vy1") == 0) return 5;
    if (strcmp(s, "vz1") == 0) return 6;
    if (strcmp(s, "ax1") == 0) return 7;
    if (strcmp(s, "ay1") == 0) return 8;
    if (strcmp(s, "az1") == 0) return 9;
    if (strcmp(s, "tension") == 0) return 10;
    if (strcmp(s, "ramp") == 0) return 11;
    if (strcmp(s, "time") == 0) return 12;
    
    return 0;
}

doublereal ModuleCatenaryLM::dGetPrivData(unsigned int i) const
{
    switch (i) {
        case 1: case 2: case 3:
            if (!virtual_nodes.empty()) {
                return virtual_nodes[0].position.dGet(i);
            }
            return 0.0;
            
        case 4: case 5: case 6:
            if (!virtual_nodes.empty()) {
                return virtual_nodes[0].velocity.dGet(i-3);
            }
            return 0.0;
            
        case 7: case 8: case 9:
            if (!virtual_nodes.empty()) {
                return virtual_nodes[0].acceleration.dGet(i-6);
            }
            return 0.0;
            
        case 10:
            // 第1セグメントの張力計算
            if (!virtual_nodes.empty()) {
                const Vec3& fairlead_pos = g_pNode->GetXCurr();
                const Vec3& fairlead_vel = g_pNode->GetVCurr();
                doublereal segment_length = L / static_cast<doublereal>(NUM_SEGMENTS);
                Vec3 F_axial = ComputeAxialForce(fairlead_pos, virtual_nodes[0].position,
                                               fairlead_vel, virtual_nodes[0].velocity, segment_length);
                return F_axial.Norm();
            }
            return 0.0;
            
        case 11:
            return GetRampFactor(simulation_time);
            
        case 12:
            return simulation_time;
            
        default:
            return 0.0;
    }
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
