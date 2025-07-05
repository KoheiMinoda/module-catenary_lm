/* MBDyn Enhanced Catenary Module with Lumped Mass Method */

#include "mbconfig.h"
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
#include "module-catenary_lm.h"

// 流体力パラメータ
static const doublereal RHO_WATER = 1025.0;
static const doublereal KINEMATIC_VISCOSITY = 1.35e-6;
static const doublereal CD_NORMAL = 1.2;
static const doublereal CD_AXIAL = 0.008;
static const doublereal DIAM_DRAG_NORMAL = 0.05;
static const doublereal DIAM_DRAG_AXIAL = 0.01592;
static const doublereal CA_NORMAL_X = 1.0;
static const doublereal CA_NORMAL_Y = 1.0;
static const doublereal CA_AXIAL_Z = 0.0;
static const doublereal CM_NORMAL_X = 2.0;
static const doublereal CM_NORMAL_Y = 2.0;
static const doublereal CM_AXIAL_Z = 1.0;

// 海流・波浪パラメータ
static const doublereal CURRENT_SURFACE = 0.0;
static const doublereal CURRENT_BOTTOM = 0.0;
static const doublereal CURRENT_DIRECTION = 0.0;
static const doublereal WAVE_HEIGHT = 0.0;
static const doublereal WAVE_PERIOD = 10.0;
static const doublereal WAVE_DIRECTION = 0.0;
static const doublereal WATER_DEPTH = 320.0;

// 数値計算制限
static const doublereal MAX_VELOCITY = 1000.0;
static const doublereal MAX_FORCE = 100.0e9;
static const doublereal MAX_ACC = 50.0;

struct VirtualNode {
    Vec3 position;
    Vec3 velocity;
    Vec3 acceleration;
    doublereal mass;
    doublereal effective_mass_x, effective_mass_y, effective_mass_z;
    bool active;
    
    VirtualNode() : 
        position(0.0, 0.0, 0.0), 
        velocity(0.0, 0.0, 0.0),
        acceleration(0.0, 0.0, 0.0),
        mass(0.0), 
        effective_mass_x(0.0), effective_mass_y(0.0), effective_mass_z(0.0),
        active(true) 
    {}
    
    VirtualNode(const Vec3& pos, doublereal m) : 
        position(pos), 
        velocity(0.0, 0.0, 0.0),
        acceleration(0.0, 0.0, 0.0),
        mass(m), 
        effective_mass_x(m), effective_mass_y(m), effective_mass_z(m),
        active(true) 
    {}
};

class ModuleCatenaryLM : virtual public Elem, public UserDefinedElem {
public:
    ModuleCatenaryLM(unsigned uLabel, const DofOwner *pDO, DataManager* pDM, MBDynParser& HP);
    virtual ~ModuleCatenaryLM(void);

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

    unsigned int iGetNumPrivData(void) const;
    int iGetNumConnectedNodes(void) const;
    void GetConnectedNodes(std::vector<const Node *>& connectedNodes) const;
    void SetValue(DataManager *pDM, VectorHandler& X, VectorHandler& XP, SimulationEntity::Hints *ph);
    std::ostream& Restart(std::ostream& out) const;

    virtual unsigned int iGetInitialNumDof(void) const;
    virtual void InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const;
    VariableSubMatrixHandler& InitialAssJac(VariableSubMatrixHandler& WorkMat, const VectorHandler& XCurr);
    SubVectorHandler& InitialAssRes(SubVectorHandler& WorkVec, const VectorHandler& XCurr);

private:
    const StructNode* g_pNode;
    double APx, APy, APz;
    double L_orig;
    doublereal rho_line;
    double xacc;
    doublereal EA;
    doublereal CA;
    doublereal line_diameter;
    doublereal g_gravity;
    doublereal seabed_z;
    doublereal K_seabed;
    doublereal C_seabed;
    doublereal MU_LATERAL;
    doublereal MU_AXIAL_STATIC;
    doublereal V_SLIP_TOL_LATERAL;
    doublereal SMOOTH_CLEARANCE;
    doublereal SMOOTH_EPS;
    doublereal CONTACT_DIAMETER;
    doublereal STRUCT_DAMP_RATIO;
    doublereal POISSON_RATIO;
    doublereal P_ATMOSPHERIC;
    doublereal RAYLEIGH_ALPHA;
    doublereal RAYLEIGH_BETA;
    doublereal simulation_time;
    doublereal prev_time;
    doublereal ramp_time;
    
    DriveOwner FSF;
    std::vector<VirtualNode> virtual_nodes;
    static const unsigned int NUM_SEGMENTS = 20;
    
    // カテナリー理論関数
    static double myasinh(double X);
    static double myacosh(double X);
    static void funcd(double x, double xacc, double &f, double &df, double d, double l, double &p0);
    static double rtsafe(double x1, double x2, double xacc, double d, double l, double &p0);
    
    void InitializeVirtualNodesFromCatenary();
    doublereal GetRampFactor(doublereal current_time) const;
    void UpdateVirtualNodesImplicit(doublereal dt, doublereal dCoef);
    
    Vec3 ComputeAxialForceImplicit(const Vec3& pos1, const Vec3& pos2, const Vec3& vel1, const Vec3& vel2, doublereal L0, doublereal dCoef) const;
    Vec3 GetCurrentVelocity(doublereal z_coord, doublereal t) const;
    void GetWaveVelocityAcceleration(doublereal x, doublereal z, doublereal t, Vec3& velocity, Vec3& acceleration) const;
    Vec3 ComputeFroudeKrylovForce(const Vec3& seg_vector, doublereal diameter, const Vec3& fluid_acc) const;
    Vec3 ComputeAddedMassForce(const Vec3& seg_vector, doublereal seg_length, doublereal diameter, const Vec3& structure_acc, const Vec3& fluid_acc) const;
    Vec3 ComputeDragForceAdvanced(const Vec3& seg_vector, doublereal seg_length, doublereal diam_normal, doublereal diam_axial, const Vec3& rel_vel, doublereal cd_normal, doublereal cd_axial) const;
    Vec3 ComputeMorisonForcesImplicit(const Vec3& pos1, const Vec3& pos2, const Vec3& vel1, const Vec3& vel2, const Vec3& acc1, const Vec3& acc2, doublereal seg_length, doublereal t, doublereal dCoef) const;
    Vec3 ComputeRayleighDampingForcesImplicit(unsigned int node_idx, const Vec3& node_vel, doublereal dCoef) const;
    
    void CalculateDirectionalEffectiveMasses();
    void CalculateSegmentAddedMassMatrix(const Vec3& seg_vector, doublereal seg_length, doublereal M_added[3][3]) const;
    doublereal CalculateExternalPressure(doublereal z_coord) const;
    void GetLocalCoordinateSystem(const Vec3& seg_vector, Vec3& t_vec, Vec3& n1_vec, Vec3& n2_vec) const;
    doublereal SmoothSeabedForce(doublereal raw_penetration, doublereal K, doublereal seg_length, doublereal radius, doublereal clearance, doublereal eps) const;
    doublereal GetCurrentTime() const;
    doublereal GetSeabedZ(doublereal x_coord) const;
    doublereal GetSeabedPenetration(const Vec3& node_pos, doublereal diameter) const;
    bool IsSegmentOnSeabed(const Vec3& pos1, const Vec3& pos2, doublereal diameter) const;
    Vec3 ComputeSegmentSeabedForces(const Vec3& pos1, const Vec3& pos2, const Vec3& vel1, const Vec3& vel2, doublereal seg_length, doublereal dCoef) const;
    bool CheckNumericalStability() const;
};

// コンストラクタ
ModuleCatenaryLM::ModuleCatenaryLM(
    unsigned uLabel,
    const DofOwner *pDO, 
    DataManager* pDM, 
    MBDynParser& HP
)
    : Elem(uLabel, flag(0)), UserDefinedElem(uLabel, pDO), 
        g_pNode(0), 
        APx(0), APy(0), APz(0),
        L_orig(0), 
        rho_line(77.71), 
        xacc(1e-4),
        EA(3.842e8), CA(0.0),
        line_diameter(0.09017),
        g_gravity(9.80665),
        seabed_z(-320.0), K_seabed(1.0e6), C_seabed(0.0),
        MU_LATERAL(0.0), MU_AXIAL_STATIC(0.0), V_SLIP_TOL_LATERAL(1.0e-5),
        SMOOTH_CLEARANCE(0.01), SMOOTH_EPS(0.01),
        CONTACT_DIAMETER(0.18),
        STRUCT_DAMP_RATIO(0.000), 
        POISSON_RATIO(0.0), 
        P_ATMOSPHERIC(101325.0),
        RAYLEIGH_ALPHA(0.0), RAYLEIGH_BETA(0.0),
        simulation_time(0.0), prev_time(0.0), ramp_time(10.0)
{
    if (HP.IsKeyWord("help")) {
        silent_cout(
            "\n"
            "Module: ModuleCatenaryLM (Enhanced Lumped Mass Method)\n"
            "Usage: catenary_lm, fairlead_node_label, \n"
            "           LineLength, total_length,\n"
            "           LineWeight, unit_weight,\n"
            "           Xacc, rtsafe_accuracy,\n"
            "           APx, APy, APz,\n"
            "           EA, axial_stiffness,\n"
            "           CA, axial_damping,\n"
            "         [ line_diameter, diameter, ]\n"
            "         [ gravity, g_acceleration, ]\n"
            "         [ seabed, z_coordinate, k_stiffness, c_damping, ]\n"
            "         [ ramp_time, ramp_duration, ]\n"
            "         [ force scale factor, (DriveCaller), ]\n"
            "         [ output, (FLAG) ] ;\n"
            "\n"
            << std::endl
        );

        if (!HP.IsArg()) {
            throw NoErr(MBDYN_EXCEPT_ARGS);
        }
    }

    g_pNode = dynamic_cast<const StructNode *>(pDM->ReadNode(HP, Node::STRUCTURAL));
    L_orig = HP.GetReal();
    rho_line = HP.GetReal();
    xacc = HP.GetReal();
    APx = HP.GetReal();
    APy = HP.GetReal();
    APz = HP.GetReal();
    EA = HP.GetReal();
    CA = HP.GetReal();

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

    if (HP.IsKeyWord("Force" "scale" "factor")) {
        FSF.Set(HP.GetDriveCaller());
    } else {
        FSF.Set(new OneDriveCaller);
    }

    SetOutputFlag(pDM->fReadOutput(HP, Elem::LOADABLE));

    virtual_nodes.resize(NUM_SEGMENTS - 1);
    doublereal segment_mass = (rho_line * L_orig) / static_cast<doublereal>(NUM_SEGMENTS);
    
    for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
        virtual_nodes[i].mass = segment_mass;
        virtual_nodes[i].effective_mass_x = segment_mass;
        virtual_nodes[i].effective_mass_y = segment_mass;
        virtual_nodes[i].effective_mass_z = segment_mass;
        virtual_nodes[i].active = true;
        virtual_nodes[i].position = Vec3(0.0, 0.0, 0.0);
        virtual_nodes[i].velocity = Vec3(0.0, 0.0, 0.0);
        virtual_nodes[i].acceleration = Vec3(0.0, 0.0, 0.0);
    }

    try {
        InitializeVirtualNodesFromCatenary();
        CalculateDirectionalEffectiveMasses();
    } catch (const std::exception& e) {
        const Vec3& FP = g_pNode->GetXCurr();
        Vec3 AP(APx, APy, APz);
        
        for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
            doublereal ratio = static_cast<doublereal>(i + 1) / static_cast<doublereal>(NUM_SEGMENTS);
            virtual_nodes[i].position = AP + (FP - AP) * ratio;
            
            doublereal sag = 0.05 * L_orig * ratio * (1.0 - ratio);
            virtual_nodes[i].position += Vec3(0.0, 0.0, -sag);
        }
        
        CalculateDirectionalEffectiveMasses();
    }

    pDM->GetLogFile() << "catenary_lm: " << uLabel << " "
        << "Segments: " << NUM_SEGMENTS << " "
        << "VirtualNodes: " << virtual_nodes.size() << " "
        << "EA: " << EA << " "
        << "Enhanced fluid forces enabled" << std::endl;
}

ModuleCatenaryLM::~ModuleCatenaryLM(void) {
    NO_OP;
}

// カテナリー理論関数
double ModuleCatenaryLM::myasinh(double X) {
    return std::log(X + std::sqrt(X * X + 1));
}

double ModuleCatenaryLM::myacosh(double X) {
    return std::log(X + std::sqrt(X + 1) * std::sqrt(X - 1));
}

void ModuleCatenaryLM::funcd(
    double x, double xacc, double& f, double& df, double d, double l, double& p0
) {
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
        throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
    }
}

double ModuleCatenaryLM::rtsafe(
    double x1, double x2, double xacc, double d, double l, double &p0
) {
    const int MAXIT = 1000;
    int j;
    double fh, fl, xh, xl, df;
    double dx, dxold, f, temp, rts;
    double p1, p2;

    funcd(x1, xacc, fl, df, d, l, p1);
    funcd(x2, xacc, fh, df, d, l, p2);

    if((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0)) {
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

    throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
}

// カテナリー理論による仮想ノード初期化
void ModuleCatenaryLM::InitializeVirtualNodesFromCatenary() {
    if (g_pNode == nullptr || virtual_nodes.empty()) {
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

    if (h > 1e-6 && L_APFP > 1e-6 && L_orig > 1e-6) {
        try {
            doublereal L0_APFP = L_orig - h;
            doublereal delta = L_APFP - L0_APFP;
            doublereal d = delta / h;
            doublereal l = L_orig / h;
            
            if (d > 0 && d < (std::sqrt(l*l - 1) - (l - 1))) {
                doublereal p0 = 0.0;
                doublereal x1 = 0.0;
                doublereal x2 = 1.0e6;
                
                doublereal Ans_x = rtsafe(x1, x2, xacc, d, l, p0);
                doublereal a = Ans_x * h;
                doublereal segment_length = L_orig / static_cast<doublereal>(NUM_SEGMENTS);
                
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

    if (!catenary_success) {
        for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
            doublereal ratio = static_cast<doublereal>(i + 1) / static_cast<doublereal>(NUM_SEGMENTS);
            virtual_nodes[i].position = AP + (FP - AP) * ratio;
            
            doublereal sag = 0.1 * L_orig * ratio * (1.0 - ratio);
            virtual_nodes[i].position += Vec3(0.0, 0.0, -sag);
            
            virtual_nodes[i].velocity = Vec3(0.0, 0.0, 0.0);
            virtual_nodes[i].active = true;
        }
    }
}

// 流体速度取得
Vec3 ModuleCatenaryLM::GetCurrentVelocity(doublereal z_coord, doublereal t) const {
    doublereal depth_ratio = std::abs(z_coord) / WATER_DEPTH;
    depth_ratio = std::min(1.0, std::max(0.0, depth_ratio));
    
    doublereal current_magnitude = CURRENT_SURFACE * (1.0 - depth_ratio) + CURRENT_BOTTOM * depth_ratio;
    
    doublereal current_dir_rad = CURRENT_DIRECTION * M_PI / 180.0;
    doublereal u_current = current_magnitude * std::cos(current_dir_rad);
    doublereal v_current = current_magnitude * std::sin(current_dir_rad);
    doublereal w_current = 0.0;
    
    return Vec3(u_current, v_current, w_current);
}

// 波浪速度・加速度取得
void ModuleCatenaryLM::GetWaveVelocityAcceleration(
    doublereal x, doublereal z, doublereal t, 
    Vec3& velocity, Vec3& acceleration
) const {
    doublereal wave_length = (g_gravity * WAVE_PERIOD * WAVE_PERIOD) / (2.0 * M_PI);
    doublereal k = 2.0 * M_PI / wave_length;
    doublereal omega = 2.0 * M_PI / WAVE_PERIOD;
    doublereal amplitude = WAVE_HEIGHT / 2.0;
    
    doublereal wave_dir_rad = WAVE_DIRECTION * M_PI / 180.0;
    doublereal kx = k * std::cos(wave_dir_rad);
    doublereal ky = k * std::sin(wave_dir_rad);
    
    doublereal phase = kx * x + ky * 0.0 - omega * t;
    
    doublereal depth_from_surface = std::abs(z);
    doublereal cosh_factor, sinh_factor;
    
    if (depth_from_surface >= WATER_DEPTH) {
        cosh_factor = std::cosh(k * WATER_DEPTH) / std::sinh(k * WATER_DEPTH);
        sinh_factor = 1.0 / std::sinh(k * WATER_DEPTH);
    } else {
        cosh_factor = std::cosh(k * (WATER_DEPTH - depth_from_surface)) / std::sinh(k * WATER_DEPTH);
        sinh_factor = std::sinh(k * (WATER_DEPTH - depth_from_surface)) / std::sinh(k * WATER_DEPTH);
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

// ローカル座標系取得
void ModuleCatenaryLM::GetLocalCoordinateSystem(
    const Vec3& seg_vector, 
    Vec3& t_vec, Vec3& n1_vec, Vec3& n2_vec
) const {
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

// フルード・クリロフ力
Vec3 ModuleCatenaryLM::ComputeFroudeKrylovForce(
    const Vec3& seg_vector, doublereal diameter, const Vec3& fluid_acc
) const {
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

    Vec3 F_FK_x = n1_vec * (RHO_WATER * volume * CM_NORMAL_X * fluid_acc_normal_x);
    Vec3 F_FK_y = n2_vec * (RHO_WATER * volume * CM_NORMAL_Y * fluid_acc_normal_y);
    Vec3 F_FK_z = t_vec * (RHO_WATER * volume * CM_AXIAL_Z * fluid_acc_axial_scalar);

    return F_FK_x + F_FK_y + F_FK_z;
}

// 付加質量力
Vec3 ModuleCatenaryLM::ComputeAddedMassForce(
    const Vec3& seg_vector, doublereal seg_length, doublereal diameter, 
    const Vec3& structure_acc, const Vec3& fluid_acc
) const {
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
    
    Vec3 F_AM_x = n1_vec * (-RHO_WATER * volume * CA_NORMAL_X * acc_normal_x);
    Vec3 F_AM_y = n2_vec * (-RHO_WATER * volume * CA_NORMAL_Y * acc_normal_y);
    Vec3 F_AM_z = t_vec * (-RHO_WATER * volume * CA_AXIAL_Z * acc_rel_axial_scalar);

    return F_AM_x + F_AM_y + F_AM_z;
}

// 抗力
Vec3 ModuleCatenaryLM::ComputeDragForceAdvanced(
    const Vec3& seg_vector, doublereal seg_length, doublereal diam_normal, 
    doublereal diam_axial, const Vec3& rel_vel, doublereal cd_normal, doublereal cd_axial
) const {
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
        Fd_n = vel_n * (0.5 * RHO_WATER * cd_normal * area_n * mag_n);
    }
    
    Vec3 Fd_t(0.0, 0.0, 0.0);
    if (mag_t > 1e-6) {
        Fd_t = vel_t * (0.5 * RHO_WATER * cd_axial * area_t * mag_t);
    }
    
    return Fd_n + Fd_t;
}

// モリソン
Vec3 ModuleCatenaryLM::ComputeMorisonForcesImplicit(
    const Vec3& pos1, const Vec3& pos2, 
    const Vec3& vel1, const Vec3& vel2, 
    const Vec3& acc1, const Vec3& acc2, 
    doublereal seg_length, doublereal t, doublereal dCoef
) const {
    Vec3 pos_mid = (pos1 + pos2) * 0.5;
    Vec3 vel_mid = (vel1 + vel2) * 0.5;
    Vec3 acc_mid = (acc1 + acc2) * 0.5;

    Vec3 seg_vec = pos2 - pos1;
    doublereal seg_length_calc = seg_vec.Norm();
    if (seg_length_calc < 1e-9) {
        return Vec3(0.0, 0.0, 0.0);
    }
    
    Vec3 u_curr = GetCurrentVelocity(pos_mid.dGet(3), t);
    Vec3 u_wave, a_wave;
    GetWaveVelocityAcceleration(pos_mid.dGet(1), pos_mid.dGet(3), t, u_wave, a_wave);
    
    Vec3 u_fluid = u_curr + u_wave;
    Vec3 a_fluid = a_wave;
    
    Vec3 F_FK = ComputeFroudeKrylovForce(seg_vec, line_diameter, a_fluid);
    Vec3 F_AM = ComputeAddedMassForce(seg_vec, seg_length, line_diameter, acc_mid, a_fluid);

    doublereal volume = M_PI * (line_diameter/2.0) * (line_diameter/2.0) * seg_length;
    doublereal m_added_avg = RHO_WATER * volume * (CA_NORMAL_X + CA_NORMAL_Y + CA_AXIAL_Z) / 3.0;
    Vec3 F_AM_correction = acc_mid * (-m_added_avg * dCoef);
    F_AM += F_AM_correction;
    
    Vec3 u_rel = u_fluid - vel_mid;
    Vec3 F_drag = ComputeDragForceAdvanced(
        seg_vec, seg_length, DIAM_DRAG_NORMAL, DIAM_DRAG_AXIAL, 
        u_rel, CD_NORMAL, CD_AXIAL
    );
    
    return F_FK + F_AM + F_drag;
}

// 有効質量計算
void ModuleCatenaryLM::CalculateSegmentAddedMassMatrix(
    const Vec3& seg_vector, doublereal seg_length, doublereal M_added[3][3]
) const {
    doublereal volume = M_PI * (line_diameter/2.0) * (line_diameter/2.0) * seg_length;
    
    Vec3 t_vec, n1_vec, n2_vec;
    GetLocalCoordinateSystem(seg_vector, t_vec, n1_vec, n2_vec);
    
    doublereal m_local[3][3] = {
        {RHO_WATER * volume * CA_NORMAL_X, 0.0, 0.0},
        {0.0, RHO_WATER * volume * CA_NORMAL_Y, 0.0},
        {0.0, 0.0, RHO_WATER * volume * CA_AXIAL_Z}
    };
    
    doublereal R[3][3] = {
        {n1_vec.dGet(1), n2_vec.dGet(1), t_vec.dGet(1)},
        {n1_vec.dGet(2), n2_vec.dGet(2), t_vec.dGet(2)},
        {n1_vec.dGet(3), n2_vec.dGet(3), t_vec.dGet(3)}
    };
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            M_added[i][j] = 0.0;
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    M_added[i][j] += R[i][k] * m_local[k][l] * R[j][l];
                }
            }
        }
    }
}

void ModuleCatenaryLM::CalculateDirectionalEffectiveMasses() {
    if (virtual_nodes.empty()) return;
    
    const Vec3& fairlead_pos = g_pNode->GetXCurr();
    Vec3 anchor_pos(APx, APy, APz);
    
    doublereal segment_length = L_orig / static_cast<doublereal>(NUM_SEGMENTS);
    
    for (unsigned int k = 0; k < virtual_nodes.size(); ++k) {
        doublereal M_structure[3][3] = {
            {virtual_nodes[k].mass, 0.0, 0.0},
            {0.0, virtual_nodes[k].mass, 0.0},
            {0.0, 0.0, virtual_nodes[k].mass}
        };
        
        doublereal M_added_total[3][3] = {{0.0}};
        
        Vec3 pos_prev, seg_vec_prev;
        if (k == 0) {
            pos_prev = fairlead_pos;
        } else {
            pos_prev = virtual_nodes[k-1].position;
        }
        seg_vec_prev = virtual_nodes[k].position - pos_prev;
        
        doublereal M_seg_added_prev[3][3];
        CalculateSegmentAddedMassMatrix(seg_vec_prev, segment_length, M_seg_added_prev);
        
        Vec3 pos_next, seg_vec_next;
        if (k == virtual_nodes.size() - 1) {
            pos_next = anchor_pos;
        } else {
            pos_next = virtual_nodes[k+1].position;
        }
        seg_vec_next = pos_next - virtual_nodes[k].position;
        
        doublereal M_seg_added_next[3][3];
        CalculateSegmentAddedMassMatrix(seg_vec_next, segment_length, M_seg_added_next);
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                M_added_total[i][j] = 0.5 * (M_seg_added_prev[i][j] + M_seg_added_next[i][j]);
            }
        }
        
        virtual_nodes[k].effective_mass_x = M_structure[0][0] + M_added_total[0][0];
        virtual_nodes[k].effective_mass_y = M_structure[1][1] + M_added_total[1][1];
        virtual_nodes[k].effective_mass_z = M_structure[2][2] + M_added_total[2][2];
        
        if (virtual_nodes[k].effective_mass_x < 1e-6) virtual_nodes[k].effective_mass_x = virtual_nodes[k].mass + 1e-6;
        if (virtual_nodes[k].effective_mass_y < 1e-6) virtual_nodes[k].effective_mass_y = virtual_nodes[k].mass + 1e-6;
        if (virtual_nodes[k].effective_mass_z < 1e-6) virtual_nodes[k].effective_mass_z = virtual_nodes[k].mass + 1e-6;
    }
}

// ランプファクター
doublereal ModuleCatenaryLM::GetRampFactor(doublereal current_time) const {
    if (current_time <= 0.0) {
        return 0.0;
    } else if (current_time >= ramp_time) {
        return 1.0;
    } else {
        return current_time / ramp_time;
    }
}

// 軸力計算（陰解法対応）
Vec3 ModuleCatenaryLM::ComputeAxialForceImplicit(
    const Vec3& pos1, const Vec3& pos2, const Vec3& vel1, const Vec3& vel2, 
    doublereal L0, doublereal dCoef
) const {
    Vec3 dx = pos2 - pos1;
    doublereal l_current = dx.Norm();
    
    if (l_current < 1e-12) {
        return Vec3(0.0, 0.0, 0.0);
    }
    
    Vec3 t = dx / l_current;
    
    Vec3 mid_pos = (pos1 + pos2) * 0.5;
    doublereal depth = std::abs(std::min(0.0, mid_pos.dGet(3)));
    doublereal Po = P_ATMOSPHERIC + RHO_WATER * g_gravity * depth;
    doublereal Ao = M_PI * (line_diameter * line_diameter);
    doublereal pressure_term = Po * Ao;
    doublereal poisson_effect = -2.0 * POISSON_RATIO * Po * Ao;
    
    doublereal strain = (l_current - L0) / L0;
    
    doublereal F_elastic = 0.0;
    if (strain > 0.0) {
        F_elastic = EA * strain + pressure_term;
    }
    
    Vec3 dv = vel2 - vel1;
    doublereal dl_dt = dx.Dot(dv) / l_current;
    doublereal strain_rate = dl_dt / L0;
    doublereal F_strain_rate_damping = EA * STRUCT_DAMP_RATIO * strain_rate;
    
    doublereal vrel = dv.Dot(t);
    doublereal Fd = CA * vrel * (1.0 + dCoef*CA / virtual_nodes[0].mass);
    
    doublereal Fax = F_elastic + F_strain_rate_damping + Fd + poisson_effect;
    Fax = std::max(0.0, std::min(Fax, MAX_FORCE));
    
    return t * Fax;
}

// レイリー減衰力
Vec3 ModuleCatenaryLM::ComputeRayleighDampingForcesImplicit(
    unsigned int node_idx, const Vec3& node_vel, doublereal dCoef
) const {
    if (node_idx >= virtual_nodes.size()) {
        return Vec3(0.0, 0.0, 0.0);
    }
    
    doublereal alpha_implicit = RAYLEIGH_ALPHA * (1.0 + dCoef * RAYLEIGH_ALPHA);
    Vec3 f_mass_damp = node_vel * (-alpha_implicit * virtual_nodes[node_idx].mass);
    
    Vec3 f_stiff_damp(0.0, 0.0, 0.0);
    
    const Vec3& fairlead_pos = g_pNode->GetXCurr();
    const Vec3& fairlead_vel = g_pNode->GetVCurr();
    Vec3 anchor_pos(APx, APy, APz);
    Vec3 anchor_vel(0.0, 0.0, 0.0);
    
    doublereal segment_length = L_orig / static_cast<doublereal>(NUM_SEGMENTS);
    doublereal K_el = EA / segment_length;
    doublereal beta_implicit = RAYLEIGH_BETA * (1.0 + dCoef * K_el / virtual_nodes[node_idx].mass);
    
    if (node_idx == 0) {
        Vec3 seg_vec = virtual_nodes[node_idx].position - fairlead_pos;
        doublereal seg_len = seg_vec.Norm();
        if (seg_len > 1e-9) {
            Vec3 t_vec = seg_vec / seg_len;
            doublereal rel_vel_t = (virtual_nodes[node_idx].velocity - fairlead_vel).Dot(t_vec);
            doublereal Fd_scalar = RAYLEIGH_BETA * K_el * rel_vel_t;
            f_stiff_damp += t_vec * Fd_scalar;
        }
    } else {
        Vec3 seg_vec = virtual_nodes[node_idx].position - virtual_nodes[node_idx-1].position;
        doublereal seg_len = seg_vec.Norm();
        if (seg_len > 1e-9) {
            Vec3 t_vec = seg_vec / seg_len;
            doublereal rel_vel_t = (virtual_nodes[node_idx].velocity - virtual_nodes[node_idx-1].velocity).Dot(t_vec);
            doublereal Fd_scalar = RAYLEIGH_BETA * K_el * rel_vel_t;
            f_stiff_damp += t_vec * Fd_scalar;
        }
    }
    
    if (node_idx == virtual_nodes.size() - 1) {
        Vec3 seg_vec = anchor_pos - virtual_nodes[node_idx].position;
        doublereal seg_len = seg_vec.Norm();
        if (seg_len > 1e-9) {
            Vec3 t_vec = seg_vec / seg_len;
            doublereal rel_vel_t = (anchor_vel - virtual_nodes[node_idx].velocity).Dot(t_vec);
            doublereal Fd_scalar = RAYLEIGH_BETA * K_el * rel_vel_t;
            f_stiff_damp -= t_vec * Fd_scalar;
        }
    } else {
        Vec3 seg_vec = virtual_nodes[node_idx+1].position - virtual_nodes[node_idx].position;
        doublereal seg_len = seg_vec.Norm();
        if (seg_len > 1e-9) {
            Vec3 t_vec = seg_vec / seg_len;
            doublereal rel_vel_t = (virtual_nodes[node_idx+1].velocity - virtual_nodes[node_idx].velocity).Dot(t_vec);
            doublereal Fd_scalar = RAYLEIGH_BETA * K_el * rel_vel_t;
            f_stiff_damp -= t_vec * Fd_scalar;
        }
    }
    
    return f_mass_damp + f_stiff_damp;
}

// 海底関連関数
doublereal ModuleCatenaryLM::GetSeabedZ(doublereal x_coord) const {
    return seabed_z;
}

doublereal ModuleCatenaryLM::GetSeabedPenetration(const Vec3& node_pos, doublereal diameter) const {
    doublereal x = node_pos.dGet(1);
    doublereal z = node_pos.dGet(3);
    doublereal seabed_z_local = GetSeabedZ(x);
    doublereal penetration = seabed_z_local - z + CONTACT_DIAMETER;
    return std::max(0.0, penetration);
}

bool ModuleCatenaryLM::IsSegmentOnSeabed(const Vec3& pos1, const Vec3& pos2, doublereal diameter) const {
    doublereal pen1 = GetSeabedPenetration(pos1, diameter);
    doublereal pen2 = GetSeabedPenetration(pos2, diameter);
    return pen1 > 0.0 && pen2 > 0.0;
}

doublereal ModuleCatenaryLM::SmoothSeabedForce(
    doublereal raw_penetration, doublereal K, doublereal seg_length, 
    doublereal radius, doublereal clearance, doublereal eps
) const {
    doublereal p_eff = raw_penetration + clearance + line_diameter/2.0;
    if (p_eff <= 0.0) {
        return 0.0;
    }
    doublereal K_eff = K * seg_length * radius * p_eff;
    return K_eff * (std::sqrt(p_eff*p_eff + eps*eps) - eps);
}

Vec3 ModuleCatenaryLM::ComputeSegmentSeabedForces(
    const Vec3& pos1, const Vec3& pos2, const Vec3& vel1, const Vec3& vel2, 
    doublereal seg_length, doublereal dCoef
) const {
    Vec3 seg_vec = pos2 - pos1;
    doublereal seg_length_calc = seg_vec.Norm();
    if (seg_length_calc < 1e-9) {
        return Vec3(0.0, 0.0, 0.0);
    }
    
    doublereal radius = line_diameter / 2.0;
    
    doublereal seabed_z_local = GetSeabedZ(0.0);
    doublereal raw_i = seabed_z_local - pos1.dGet(3);
    doublereal raw_j = seabed_z_local - pos2.dGet(3);
    doublereal raw_avg = 0.5 * (raw_i + raw_j);
    
    doublereal F_normal = SmoothSeabedForce(
        raw_avg, K_seabed, seg_length, radius, SMOOTH_CLEARANCE, SMOOTH_EPS
    );
    if (F_normal < 0.0) {
        F_normal = 0.0;
    }

    doublereal stiffness_correction = 1.0 + dCoef * K_seabed / (virtual_nodes[0].mass + 1e-6);
    F_normal *= stiffness_correction;
    
    Vec3 pos_mid = (pos1 + pos2) * 0.5;
    Vec3 vel_mid = (vel1 + vel2) * 0.5;

    doublereal damping_correction = 1.0 + dCoef * C_seabed / (virtual_nodes[0].mass + 1e-6);
    doublereal vel_z_damped = vel_mid.dGet(3) * damping_correction;
    F_normal -= C_seabed * vel_z_damped;
    F_normal = std::max(0.0, F_normal);
    
    Vec3 vel_lat(vel_mid.dGet(1), vel_mid.dGet(2), 0.0);
    doublereal mag_lat = vel_lat.Norm();
    
    Vec3 F_fric_lat(0.0, 0.0, 0.0);
    if (mag_lat > V_SLIP_TOL_LATERAL) {
        doublereal friction_correction = 1.0 + dCoef * 0.1;
        F_fric_lat = vel_lat * (-MU_LATERAL * F_normal * friction_correction / mag_lat);
    }

    Vec3 F_norm_vec(0.0, 0.0, F_normal);
    Vec3 F_fric_vec(F_fric_lat.dGet(1), F_fric_lat.dGet(2), 0.0);
    
    return F_norm_vec + F_fric_vec;
}

// 仮想ノード更新（陰解法）
void ModuleCatenaryLM::UpdateVirtualNodesImplicit(doublereal dt, doublereal dCoef) {
    if (virtual_nodes.empty()) return;

    if (dt <= 1e-12 || dt > 0.1) {
        return;
    }

    if (g_pNode == nullptr) {
        return;
    }

    const Vec3& fairlead_pos = g_pNode->GetXCurr();
    const Vec3& fairlead_vel = g_pNode->GetVCurr();
    Vec3 anchor_pos(APx, APy, APz);
    Vec3 anchor_vel(0.0, 0.0, 0.0);
    
    doublereal segment_length = L_orig / static_cast<doublereal>(NUM_SEGMENTS);
    doublereal ramp_factor = GetRampFactor(simulation_time);
    
    CalculateDirectionalEffectiveMasses();
    
    for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
        if (!virtual_nodes[i].active) continue;
        
        Vec3 F_total(0.0, 0.0, 0.0);
        
        Vec3 F_gravity(0.0, 0.0, -virtual_nodes[i].mass * g_gravity * ramp_factor);
        F_total += F_gravity;
        
        Vec3 F_axial_prev(0.0, 0.0, 0.0);
        if (i == 0) {
            F_axial_prev = ComputeAxialForceImplicit(
                fairlead_pos, virtual_nodes[i].position, 
                fairlead_vel, virtual_nodes[i].velocity, 
                segment_length, dCoef
            );
        } else {
            F_axial_prev = ComputeAxialForceImplicit(
                virtual_nodes[i-1].position, virtual_nodes[i].position,
                virtual_nodes[i-1].velocity, virtual_nodes[i].velocity, 
                segment_length, dCoef
            );
        }
        F_total += F_axial_prev * ramp_factor;
        
        Vec3 F_axial_next(0.0, 0.0, 0.0);
        if (i == virtual_nodes.size() - 1) {
            F_axial_next = ComputeAxialForceImplicit(
                virtual_nodes[i].position, anchor_pos,
                virtual_nodes[i].velocity, anchor_vel, 
                segment_length, dCoef
            );
        } else {
            F_axial_next = ComputeAxialForceImplicit(
                virtual_nodes[i].position, virtual_nodes[i+1].position,
                virtual_nodes[i].velocity, virtual_nodes[i+1].velocity, 
                segment_length, dCoef
            );
        }
        F_total -= F_axial_next * ramp_factor;
        
        Vec3 F_fluid(0.0, 0.0, 0.0);
        
        if (i == 0) {
            F_fluid += ComputeMorisonForcesImplicit(
                fairlead_pos, virtual_nodes[i].position,
                fairlead_vel, virtual_nodes[i].velocity,
                Vec3(0.0, 0.0, 0.0), virtual_nodes[i].acceleration,
                segment_length, simulation_time, dCoef
            ) * 0.5;
        } else {
            F_fluid += ComputeMorisonForcesImplicit(
                virtual_nodes[i-1].position, virtual_nodes[i].position,
                virtual_nodes[i-1].velocity, virtual_nodes[i].velocity,
                virtual_nodes[i-1].acceleration, virtual_nodes[i].acceleration,
                segment_length, simulation_time, dCoef
            ) * 0.5;
        }
        
        if (i == virtual_nodes.size() - 1) {
            F_fluid += ComputeMorisonForcesImplicit(
                virtual_nodes[i].position, anchor_pos,
                virtual_nodes[i].velocity, anchor_vel,
                virtual_nodes[i].acceleration, Vec3(0.0, 0.0, 0.0),
                segment_length, simulation_time, dCoef
            ) * 0.5;
        } else {
            F_fluid += ComputeMorisonForcesImplicit(
                virtual_nodes[i].position, virtual_nodes[i+1].position,
                virtual_nodes[i].velocity, virtual_nodes[i+1].velocity,
                virtual_nodes[i].acceleration, virtual_nodes[i+1].acceleration,
                segment_length, simulation_time, dCoef
            ) * 0.5;
        }
        
        F_total += F_fluid * ramp_factor;
        
        if (IsSegmentOnSeabed(virtual_nodes[i].position, virtual_nodes[i].position, line_diameter)) {
            Vec3 F_seabed = ComputeSegmentSeabedForces(
                virtual_nodes[i].position, virtual_nodes[i].position,
                virtual_nodes[i].velocity, virtual_nodes[i].velocity,
                segment_length, dCoef
            );
            F_total += F_seabed * ramp_factor;
        }
        
        Vec3 F_rayleigh = ComputeRayleighDampingForcesImplicit(
            i, virtual_nodes[i].velocity, dCoef
        );
        F_total += F_rayleigh;
        
        doublereal eff_mass_x = virtual_nodes[i].effective_mass_x * (1.0 + dCoef * RAYLEIGH_ALPHA);
        doublereal eff_mass_y = virtual_nodes[i].effective_mass_y * (1.0 + dCoef * RAYLEIGH_ALPHA);
        doublereal eff_mass_z = virtual_nodes[i].effective_mass_z * (1.0 + dCoef * RAYLEIGH_ALPHA);

        Vec3 acceleration_new(
            F_total.dGet(1) / eff_mass_x,
            F_total.dGet(2) / eff_mass_y,
            F_total.dGet(3) / eff_mass_z
        );
        
        doublereal acc_norm = acceleration_new.Norm();
        doublereal max_acc = MAX_ACC * g_gravity;
        if (acc_norm > max_acc) {
            acceleration_new = acceleration_new * (max_acc / acc_norm);
        }
        
        virtual_nodes[i].acceleration = acceleration_new;
        virtual_nodes[i].velocity += virtual_nodes[i].acceleration * dt;
        
        doublereal vel_norm = virtual_nodes[i].velocity.Norm();
        if (vel_norm > MAX_VELOCITY) {
            virtual_nodes[i].velocity = virtual_nodes[i].velocity * (MAX_VELOCITY / vel_norm);
        }
        
        virtual_nodes[i].position += virtual_nodes[i].velocity * dt;
    }
}

// 数値安定性チェック
bool ModuleCatenaryLM::CheckNumericalStability() const {
    for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
        if (!virtual_nodes[i].active) continue;
        
        Vec3 pos = virtual_nodes[i].position;
        if (!std::isfinite(pos.dGet(1)) || !std::isfinite(pos.dGet(2)) || !std::isfinite(pos.dGet(3))) {
            return false;
        }
        
        Vec3 vel = virtual_nodes[i].velocity;
        doublereal vel_norm = vel.Norm();
        if (!std::isfinite(vel_norm) || vel_norm > MAX_VELOCITY * 2.0) {
            return false;
        }
        
        Vec3 acc = virtual_nodes[i].acceleration;
        doublereal acc_norm = acc.Norm();
        if (!std::isfinite(acc_norm) || acc_norm > MAX_ACC * g_gravity * 2.0) {
            return false;
        }
        
        if (virtual_nodes[i].effective_mass_x <= 0.0 || 
            virtual_nodes[i].effective_mass_y <= 0.0 || 
            virtual_nodes[i].effective_mass_z <= 0.0) {
            return false;
        }
        
        if (!std::isfinite(virtual_nodes[i].effective_mass_x) ||
            !std::isfinite(virtual_nodes[i].effective_mass_y) ||
            !std::isfinite(virtual_nodes[i].effective_mass_z)) {
            return false;
        }
    }
    
    if (g_pNode != nullptr && !virtual_nodes.empty()) {
        const Vec3& fairlead_pos = g_pNode->GetXCurr();
        Vec3 anchor_pos(APx, APy, APz);
        
        Vec3 seg1 = virtual_nodes[0].position - fairlead_pos;
        doublereal len1 = seg1.Norm();
        doublereal expected_len = L_orig / static_cast<doublereal>(NUM_SEGMENTS);
        
        if (len1 > expected_len * 5.0 || len1 < expected_len * 0.2) {
            return false;
        }
        
        Vec3 seg_last = anchor_pos - virtual_nodes.back().position;
        doublereal len_last = seg_last.Norm();
        
        if (len_last > expected_len * 5.0 || len_last < expected_len * 0.2) {
            return false;
        }
    }
    
    if (!std::isfinite(simulation_time) || simulation_time < 0.0) {
        return false;
    }
    
    return true;
}

// ユーティリティ関数
doublereal ModuleCatenaryLM::CalculateExternalPressure(doublereal z_coord) const {
    doublereal depth = std::abs(std::min(0.0, z_coord));
    return P_ATMOSPHERIC + RHO_WATER * g_gravity * depth;
}

doublereal ModuleCatenaryLM::GetCurrentTime() const {
    return simulation_time;
}

// AssRes関数：残差ベクトル計算（メインの力計算）
SubVectorHandler& ModuleCatenaryLM::AssRes(
    SubVectorHandler& WorkVec,
    doublereal dCoef,
    const VectorHandler& XCurr, 
    const VectorHandler& XPrimeCurr
) {
    integer iNumRows = 6;
    WorkVec.ResizeReset(iNumRows);

    integer iFirstMomIndex = g_pNode->iGetFirstMomentumIndex();
    
    for (int iCnt = 1; iCnt <= 6; iCnt++) {
        WorkVec.PutRowIndex(iCnt, iFirstMomIndex + iCnt);
    }

    doublereal current_time = GetCurrentTime();
    doublereal dt = current_time - prev_time;
    
    if (dt > 1e-12 && dt < 1.0) {
        simulation_time = current_time;
        UpdateVirtualNodesImplicit(dt, dCoef);
        prev_time = current_time;
    }

    doublereal dFSF = FSF.dGet();
    doublereal ramp_factor = GetRampFactor(current_time);
    
    Vec3 F_total(0.0, 0.0, 0.0);
    Vec3 M_total(0.0, 0.0, 0.0);
    
    if (!virtual_nodes.empty() && virtual_nodes[0].active) {
        
        const Vec3& fairlead_pos = g_pNode->GetXCurr();
        const Vec3& fairlead_vel = g_pNode->GetVCurr();

        doublereal segment_length = L_orig / static_cast<doublereal>(NUM_SEGMENTS);
        
        // フェアリーダーから最初の仮想ノードへの軸力
        Vec3 F_axial = ComputeAxialForceImplicit(
            fairlead_pos, virtual_nodes[0].position,
            fairlead_vel, virtual_nodes[0].velocity, 
            segment_length, 
            dCoef
        );
        
        // フェアリーダーセグメントの流体力
        Vec3 F_fluid = ComputeMorisonForcesImplicit(
            fairlead_pos, virtual_nodes[0].position,
            fairlead_vel, virtual_nodes[0].velocity,
            Vec3(0.0, 0.0, 0.0), virtual_nodes[0].acceleration,
            segment_length, current_time, dCoef
        ) * 0.5;
        
        // 海底接触力
        Vec3 F_seabed(0.0, 0.0, 0.0);
        if (IsSegmentOnSeabed(fairlead_pos, virtual_nodes[0].position, line_diameter)) {
            F_seabed = ComputeSegmentSeabedForces(
                fairlead_pos, fairlead_pos,
                fairlead_vel, fairlead_vel,
                segment_length * 0.5, dCoef
            );
        }

        const Vec3& fairlead_acc = g_pNode->GetXPPCurr();
        doublereal seg_mass = rho_line * segment_length * 0.5;
        Vec3 F_inertial = fairlead_acc * (-seg_mass * ramp_factor);
        
        // 総合力の計算
        F_total = (F_axial + F_fluid + F_seabed + F_inertial) * ramp_factor;
        
        // モーメントの計算
        Vec3 r_vec = virtual_nodes[0].position - fairlead_pos;
        doublereal r_norm = r_vec.Norm();
        if (r_norm > 1e-6) {
            M_total = r_vec.Cross(F_axial) * ramp_factor;
            Vec3 M_fluid = r_vec.Cross(F_axial) * 0.1 * ramp_factor;
            M_total += M_fluid;
        } 
        
        // 安全性チェック
        doublereal force_norm = F_total.Norm();
        if (force_norm > MAX_FORCE) {
            F_total = F_total * (MAX_FORCE / force_norm);
        }
        
        doublereal moment_norm = M_total.Norm();
        doublereal max_moment = 1.0e6;
        if (moment_norm > max_moment) {
            M_total = M_total * (max_moment / moment_norm);
        }

        static Vec3 F_prev(0.0, 0.0, 0.0);
        static Vec3 M_prev(0.0, 0.0, 0.0);
        
        F_prev = F_total;
        M_prev = M_total;
    }
    
    // Force Scale Factor の適用
    F_total *= dFSF;
    M_total *= dFSF;
    
    // MBDynの残差ベクトルへの追加
    WorkVec.Add(1, F_total);
    WorkVec.Add(4, M_total);

    return WorkVec;
}

// WorkSpaceDim: 作業空間サイズ定義
void ModuleCatenaryLM::WorkSpaceDim(integer* piNumRows, integer* piNumCols) const {
    *piNumRows = 6;
    *piNumCols = 6;
}

// Output: 結果出力
void ModuleCatenaryLM::Output(OutputHandler& OH) const {
    if (bToBeOutput()) {
        if (OH.UseText(OutputHandler::LOADABLE)) {
            const Vec3& FP = g_pNode->GetXCurr();
            doublereal current_time = GetCurrentTime();
            
            OH.Loadable() << GetLabel()
                << " " << current_time
                << " " << FP.dGet(1) // フェアリーダー位置
                << " " << FP.dGet(2)
                << " " << FP.dGet(3)
                << " " << virtual_nodes.size()  // 仮想ノード数
                << " " << GetRampFactor(simulation_time) // ランプファクター
                << " " << FSF.dGet(current_time)
                << std::endl;
            
            // 詳細な仮想ノード情報も出力
            for (unsigned int i = 0; i < virtual_nodes.size(); ++i) {
                if (virtual_nodes[i].active) {
                    OH.Loadable() << " vnode_" << i 
                        << " " << virtual_nodes[i].position.dGet(1)
                        << " " << virtual_nodes[i].position.dGet(2)
                        << " " << virtual_nodes[i].position.dGet(3)
                        << " " << virtual_nodes[i].velocity.Norm()
                        << " " << virtual_nodes[i].acceleration.Norm()
                        << std::endl;
                }
            }
        }
    }
}

// MBDynインターフェース関数群
unsigned int ModuleCatenaryLM::iGetNumPrivData(void) const {
    return 0;
}

int ModuleCatenaryLM::iGetNumConnectedNodes(void) const {
    return 1;
}

void ModuleCatenaryLM::GetConnectedNodes(std::vector<const Node *>& connectedNodes) const {
    connectedNodes.resize(1);
    connectedNodes[0] = g_pNode;
}

void ModuleCatenaryLM::SetValue(DataManager *pDM, VectorHandler& X, VectorHandler& XP, SimulationEntity::Hints *ph) {
    NO_OP;
}

std::ostream& ModuleCatenaryLM::Restart(std::ostream& out) const {
    return out << "# ModuleCatenaryLM: not implemented" << std::endl;
}

unsigned int ModuleCatenaryLM::iGetInitialNumDof(void) const {
    return 0;
}

void ModuleCatenaryLM::InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const {
    *piNumRows = 0;
    *piNumCols = 0;
}

VariableSubMatrixHandler& ModuleCatenaryLM::InitialAssJac(
    VariableSubMatrixHandler& WorkMat, 
    const VectorHandler& XCurr
) {
    ASSERT(0);
    WorkMat.SetNullMatrix();
    return WorkMat;
}

SubVectorHandler& ModuleCatenaryLM::InitialAssRes(
    SubVectorHandler& WorkVec, 
    const VectorHandler& XCurr
) {
    ASSERT(0);
    WorkVec.ResizeReset(0);
    return WorkVec;
}

// AssJac関数：ヤコビアン行列計算（陰解法対応）
VariableSubMatrixHandler& ModuleCatenaryLM::AssJac(
    VariableSubMatrixHandler& WorkMat,
    doublereal dCoef, 
    const VectorHandler& XCurr,
    const VectorHandler& XPrimeCurr
) {
    integer iNumRows = 0;
    integer iNumCols = 0;
    WorkSpaceDim(&iNumRows, &iNumCols);
    
    FullSubMatrixHandler& WM = WorkMat.SetFull();
    WM.ResizeReset(iNumRows, iNumCols);

    integer iFirstPosIndex = g_pNode->iGetFirstPositionIndex();
    integer iFirstMomIndex = g_pNode->iGetFirstMomentumIndex();
    
    for (int iCnt = 1; iCnt <= 6; iCnt++) {
        WM.PutRowIndex(iCnt, iFirstMomIndex + iCnt);
        WM.PutColIndex(iCnt, iFirstPosIndex + iCnt);
    }

    const Vec3& fairlead_pos = g_pNode->GetXCurr();
    const Vec3& fairlead_vel = g_pNode->GetVCurr();
    
    if (virtual_nodes.empty() || !virtual_nodes[0].active) {
        for (int i = 1; i <= 6; i++) {
            for (int j = 1; j <= 6; j++) {
                WM.PutCoef(i, j, 0.0);
            }
        }
        return WorkMat;
    }

    doublereal segment_length = L_orig / static_cast<doublereal>(NUM_SEGMENTS);
    doublereal ramp_factor = GetRampFactor(GetCurrentTime());
    
    // ========= 位置に対する偏微分（K行列成分） =========
    Vec3 seg_vec = virtual_nodes[0].position - fairlead_pos;
    doublereal seg_len = seg_vec.Norm();
    
    if (seg_len > 1e-9) {
        Vec3 t_vec = seg_vec / seg_len;
        
        doublereal strain = (seg_len - segment_length) / segment_length;
        doublereal K_tangent = 0.0;
        
        if (strain > 0.0) {
            K_tangent = EA / segment_length;
            doublereal F_axial_current = EA * strain;
            doublereal K_geometric = F_axial_current / seg_len;
            
            for (int i = 1; i <= 3; i++) {
                for (int j = 1; j <= 3; j++) {
                    doublereal delta_ij = (i == j) ? 1.0 : 0.0;
                    
                    doublereal K_material = K_tangent * t_vec.dGet(i) * t_vec.dGet(j);
                    doublereal K_geom = K_geometric * (delta_ij - t_vec.dGet(i) * t_vec.dGet(j));
                    
                    doublereal K_total = (K_material + K_geom) * ramp_factor;
                    
                    WM.PutCoef(i, j, -K_total);
                }
            }
        }
    }
    
    // 流体力の線形化項（付加質量による剛性効果）
    doublereal volume = M_PI * (line_diameter/2.0) * (line_diameter/2.0) * segment_length;
    doublereal m_added_avg = RHO_WATER * volume * (CA_NORMAL_X + CA_NORMAL_Y + CA_AXIAL_Z) / 3.0;
    doublereal fluid_stiffness = m_added_avg * ramp_factor * 0.1;
    
    for (int i = 1; i <= 3; i++) {
        WM.IncCoef(i, i, -fluid_stiffness);
    }
    
    // ========= 速度に対する偏微分（C行列成分） =========
    if (seg_len > 1e-9) {
        Vec3 t_vec = seg_vec / seg_len;
        
        doublereal C_axial = CA * ramp_factor;
        doublereal C_structural = EA * STRUCT_DAMP_RATIO / segment_length * ramp_factor;
        doublereal C_total = C_axial + C_structural;
        
        doublereal C_implicit = C_total * (1.0 + dCoef * C_total / virtual_nodes[0].mass);
        
        for (int i = 1; i <= 3; i++) {
            for (int j = 1; j <= 3; j++) {
                doublereal C_ij = C_implicit * t_vec.dGet(i) * t_vec.dGet(j);
                WM.PutCoef(i, j + 3, -C_ij * dCoef);
            }
        }
    }
    
    // 流体減衰の線形化
    Vec3 u_curr = GetCurrentVelocity(fairlead_pos.dGet(3), GetCurrentTime());
    Vec3 u_wave, a_wave;
    GetWaveVelocityAcceleration(
        fairlead_pos.dGet(1), fairlead_pos.dGet(3), 
        GetCurrentTime(), u_wave, a_wave
    );
    
    Vec3 u_fluid = u_curr + u_wave;
    Vec3 vel_rel = u_fluid - fairlead_vel;
    doublereal vel_rel_norm = vel_rel.Norm();
    
    if (vel_rel_norm > 1e-6) {
        doublereal area_drag = DIAM_DRAG_NORMAL * segment_length;
        doublereal drag_coeff = 0.5 * RHO_WATER * CD_NORMAL * area_drag * vel_rel_norm * ramp_factor;
        doublereal drag_implicit = drag_coeff * (1.0 + dCoef * 0.1);
        
        for (int i = 1; i <= 3; i++) {
            WM.IncCoef(i, i + 3, -drag_implicit * dCoef * 0.5);
        }
    }
    
    // レイリー減衰の寄与
    doublereal mass_damp = RAYLEIGH_ALPHA * virtual_nodes[0].mass * ramp_factor;
    doublereal stiff_damp = RAYLEIGH_BETA * EA / segment_length * ramp_factor;
    doublereal rayleigh_implicit = (mass_damp + stiff_damp) * (1.0 + dCoef * RAYLEIGH_ALPHA);
    
    for (int i = 1; i <= 3; i++) {
        WM.IncCoef(i, i + 3, -rayleigh_implicit * dCoef * 0.5);
    }
    
    // 海底接触がある場合の剛性・減衰寄与
    if (IsSegmentOnSeabed(fairlead_pos, virtual_nodes[0].position, line_diameter)) {
        doublereal seabed_stiffness = K_seabed * ramp_factor;
        doublereal seabed_damping = C_seabed * ramp_factor;
        
        doublereal seabed_stiff_implicit = seabed_stiffness * (1.0 + dCoef * seabed_stiffness / virtual_nodes[0].mass);
        doublereal seabed_damp_implicit = seabed_damping * (1.0 + dCoef * seabed_damping / virtual_nodes[0].mass);
        
        WM.IncCoef(3, 3, -seabed_stiff_implicit);
        WM.IncCoef(3, 6, -seabed_damp_implicit * dCoef);
    }
    
    // ========= 慣性項の寄与（付加質量効果） =========
    doublereal inertia_correction = m_added_avg * ramp_factor;
    doublereal inertia_implicit = inertia_correction * (1.0 + dCoef);
    
    for (int i = 1; i <= 3; i++) {
        WM.IncCoef(i, i + 3, -inertia_implicit * dCoef * dCoef);
    }
    
    // ========= Force Scale Factorの寄与 =========
    doublereal dFSF = FSF.dGet();
    
    for (int i = 1; i <= 6; i++) {
        for (int j = 1; j <= 6; j++) {
            doublereal current_val = WM.dGetCoef(i, j);
            WM.PutCoef(i, j, current_val * dFSF);
        }
    }
    
    // ========= モーメント成分のヤコビアン =========
    Vec3 r_vec = virtual_nodes[0].position - fairlead_pos;
    
    Vec3 F_axial_current = ComputeAxialForceImplicit(
        fairlead_pos, virtual_nodes[0].position,
        fairlead_vel, virtual_nodes[0].velocity,
        segment_length, dCoef
    );
    
    doublereal F_skew[3][3] = {
        { 0.0, -F_axial_current.dGet(3), F_axial_current.dGet(2)},
        { F_axial_current.dGet(3), 0.0, -F_axial_current.dGet(1)},
        {-F_axial_current.dGet(2), F_axial_current.dGet(1), 0.0}
    };
    
    doublereal r_skew[3][3] = {
        { 0.0, -r_vec.dGet(3), r_vec.dGet(2)},
        { r_vec.dGet(3), 0.0, -r_vec.dGet(1)},
        {-r_vec.dGet(2), r_vec.dGet(1), 0.0}
    };
    
    for (int i = 4; i <= 6; i++) {
        for (int j = 1; j <= 3; j++) {
            WM.PutCoef(i, j, F_skew[i-4][j-1] * ramp_factor * 0.1 * dFSF);
        }
        for (int j = 1; j <= 3; j++) {
            doublereal moment_force_jac = r_skew[i-4][j-1] * ramp_factor * 0.1;
            
            for (int k = 1; k <= 3; k++) {
                WM.IncCoef(i, k, moment_force_jac * WM.dGetCoef(j, k));
            }
            for (int k = 4; k <= 6; k++) {
                WM.IncCoef(i, k, moment_force_jac * WM.dGetCoef(j, k));
            }
        }
    }

    return WorkMat;
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
            silent_cerr("catenary_lm: "
                "module_init(" << module_name << ") "
                "failed" << std::endl);
            return -1;
        }
        return 0;
    }
}

#endif // ! STATIC_MODULES
