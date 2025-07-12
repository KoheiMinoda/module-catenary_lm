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
#include "module-catenary.h"

class ModuleCatenary
	: virtual public Elem, public UserDefinedElem {
		public:
			ModuleCatenary(unsigned uLabel, const DofOwner *pDO, DataManager* pDM, MBDynParser& HP);
			virtual ~ModuleCatenary(void);

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

			unsigned int iGetNumPrivData(void) const;
			int iGetNumConnectedNodes(void) const;
			void GetConnectedNodes(std::vector<const Node *>& connectedNodes) const;
			void SetValue(DataManager *pDM, VectorHandler& X, VectorHandler& XP,
				SimulationEntity::Hints *ph);
			std::ostream& Restart(std::ostream& out) const;

			virtual unsigned int iGetInitialNumDof(void) const;
			virtual void InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const;
   			VariableSubMatrixHandler&
			InitialAssJac(VariableSubMatrixHandler& WorkMat, 
		    	const VectorHandler& XCurr);
   			SubVectorHandler& 
			InitialAssRes(SubVectorHandler& WorkVec, const VectorHandler& XCurr);
    			void GetNode();

		private:
    		const StructNode	*g_pNode;
			double GFPx,GFPy,GFPz;
			double APx, APy, APz;
			double FPx, FPy, FPz;
			double L;
			double S;
			double w;
			double L0_APFP;
			double L_APFP;
			double H, V;
			double dFx, dFy, dFz;
			double Fx, Fy, Fz;
			double delta;
			double x, d, l, h;
			double f, df, p0;
			double x1, x2, xacc;
			double X;

			DriveOwner FSF;
			Vec3 F;
			Vec3 M;
			Vec3 GFP;
			Vec3 AP;
			Vec3 FP_node;
			Vec3 FP;
			Vec3 FP_AP;
			Vec3 GFP_AP;

		private:
			double myasinh(double X);
			double myacosh(double X);
			double myatanh(double X);

			void funcd(double x, double xacc, double &f, double &df, double d, double l, double &p0);
			double rtsafe(double x1, double x2, double xacc, double d, double l, double &p0);
	};

ModuleCatenary::ModuleCatenary(
	unsigned uLabel,
	const DofOwner *pDO, 
	DataManager* pDM, 
	MBDynParser& HP
)
	: Elem(uLabel, flag(0)), UserDefinedElem(uLabel, pDO), 
		g_pNode(0), 
		GFPx(0), GFPy(0), GFPz(0), 
		APx(0), APy(0), APz(0),
		FPx(0), FPy(0), FPz(0),
		L(0), S(0), w(0),
		L0_APFP(0),	L_APFP(0),
		H(0), V(0),
		dFx(0), dFy(0), dFz(0),
		Fx(0), Fy(0), Fz(0),
		delta(0), x(0), d(0), l(0), h(0),
		p0(0), x1(0), x2(0), xacc(0), df(0), X(0)
	{
		if (HP.IsKeyWord("help")) {
			silent_cout(
				"\n"
				"Module: 	ModuleCatenary\n"
				"\n"
				<< std::endl
			);

			if (!HP.IsArg()) {
				throw NoErr(MBDYN_EXCEPT_ARGS);
			}
		}

		g_pNode = dynamic_cast<const StructNode *>(pDM->ReadNode(HP, Node::STRUCTURAL));
		L  =  HP.GetReal();
		w  =  HP.GetReal();
		xacc =  HP.GetReal();
		APx =  HP.GetReal();
		APy =  HP.GetReal();
		APz =  HP.GetReal();

		if (HP.IsKeyWord("Force" "scale" "factor")) {
			FSF.Set(HP.GetDriveCaller());

		} else {
			FSF.Set(new OneDriveCaller);
		}

		SetOutputFlag(pDM->fReadOutput(HP, Elem::LOADABLE));

		pDM->GetLogFile() << "catenary: "
			<< uLabel << " "
			<< std::endl;
	}

ModuleCatenary::~ModuleCatenary(void)
	{
		NO_OP;
	}

void ModuleCatenary::Output(OutputHandler& OH) const
	{
		if (bToBeOutput()) {
			if (OH.UseText(OutputHandler::LOADABLE)) {
				OH.Loadable() << GetLabel()
					<< " " << FPx
					<< " " << FPy
					<< " " << FPz
					<< " " << L_APFP
					<< " " << Fx
					<< " " << Fy
					<< " " << Fz
					<< std::endl;
			}
		}
	}

void ModuleCatenary::WorkSpaceDim(integer* piNumRows, integer* piNumCols) const 
	{
		*piNumRows = 6;
		*piNumCols = 6;
	}

VariableSubMatrixHandler& ModuleCatenary::AssJac( VariableSubMatrixHandler& WorkMat, doublereal dCoef, const VectorHandler& XCurr, const VectorHandler& XPrimeCurr )
	{
		WorkMat.SetNullMatrix();
		return WorkMat;
	}

double ModuleCatenary::myasinh(double X) 
	{
		return std::log(X + std::sqrt(X * X + 1));
	}

double ModuleCatenary::myacosh(double X) 
	{
		return std::log(X + std::sqrt(X + 1) * std::sqrt(X - 1));
	}

double ModuleCatenary::myatanh(double X)
	{
		return 0.5 * std::log((1 + X) / (1 - X));
	}

void ModuleCatenary::funcd(double x, double xacc, double& f, double& df, double d, double l, double& p0)
	{
    		int i,max;
		double f1, df1;
    		max = 1000;
    		if(x==0.0) {
        		f=-d;
        		df=0e-0;
        		p0=0e-0;
    		}
    		else if(x>0.0) {
        		if(l<=0.0) {
				double X_1;
				X_1 = 1.0/x+1.0;

				f=x*myacosh(X_1)-std::sqrt(1.0+2.0*x)+1.0-d;
				df=myacosh(X_1)-1.0/std::sqrt(1.0+2.0*x)-1.0/(x*std::sqrt(std::pow(X_1, 2.0)-1.0));
            			p0=0.0;
        		} else {
            			if(x>(l*l-1.0)/2) {
                			p0=0.0;
                			for(int i=1; i<max; i++) {
						double func1;
						func1 = 1.0/x+1.0/cos(p0);
					
						f1=x*(std::sqrt(std::pow(func1,2.0)-1.0)-std::tan(p0))-l;
						df1=x*(func1*std::tan(p0)*(1.0/cos(p0))/std::sqrt(std::pow(func1,2.0)-1.0)-std::pow(std::tan(p0), 2.0)-1.0);
                    				p0=p0-f1/df1;
						f1=x*(std::sqrt(std::pow(func1,2.0)-1.0)-std::tan(p0))-l;

                    				if(fabs(f1)<xacc) { break; }
                			}
				
					if(fabs(f1)>xacc) {
						std::cout<< "fabs(f1)>eps" << std::endl;
						throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
					}

					double X_2 = l/x+std::tan(p0);
					double X_3 = std::tan(p0);

					f=x*(myasinh(X_2)-myasinh(X_3))-l+1.0-d;
					df=myasinh(X_2)-myasinh(X_3)-l/(x*std::sqrt(std::pow(X_2, 2.0)+1.0));

				} else {
					double X_5;
					X_5 = 1.0/x+1.0;

					f=x*myacosh(X_5)-std::sqrt(1.0+2.0*x)+1.0-d;
					df=myacosh(X_5)-1.0/std::sqrt(1.0+2.0*x)-1.0/(x*std::sqrt(std::pow(X_5, 2.0)-1.0));
                			p0=0.0;
            			}
        		}
    		} else {
			std::cout << "ERROR (x<0)" << std::endl;
			throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
		}
	}

double ModuleCatenary::rtsafe(double x1, double x2, double xacc, double d, double l, double &p0)
	{
		const int MAXIT=1000;
    	int j;
		double fh,fl,xh,xl;
    	double dx,dxold,f,temp,rts;
		double p1, p2;

    	ModuleCatenary::funcd(x1, xacc, fl, df, d, l, p1);
    	ModuleCatenary::funcd(x2, xacc, fh, df, d, l, p2);

    	if((fl>0.0&&fh>0.0)||(fl<0.0&&fh<0.0)) {
			std::cout << "ERROR (fl>0.0&&fh>0.0)||(fl<0.0&&fh<0.0)" << std::endl;
			throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
		}

    	if(fl==0.0) {
			p0  = p1;
			rts = x1;
			return rts;
		}
    	if(fh==0.0) {
			p0  = p2;
			rts = x2;
			return rts;
		}

    	if(fl<0.0) {
        	xl=x1;
        	xh=x2;
    	} else {
        	xh=x1;
        	xl=x2;
    	}

    	rts=0.5*(x1+x2);
    	dxold=std::fabs(x2-x1);
    	dx=dxold;
    	ModuleCatenary::funcd(rts, xacc, f, df, d, l, p0);

    	for(j=0; j<MAXIT; j++) {
        	if((((rts-xh)*df-f)*((rts-xl)*df-f)>0.0)||((std::fabs(2.0*f))>std::fabs(dxold*df))) {
            		dxold = dx;
            		dx = 0.5*(xh-xl);
            		rts =xl+dx;	
            		if(xl==rts) { return rts; }
        	} else {
            		dxold=dx;
            		dx=f/df;
            		temp=rts;
            		rts-=dx;
            		if(temp==rts) {return rts;}
        	}
        	if(std::fabs(dx)<xacc) { return rts; }

		ModuleCatenary::funcd(rts, xacc, f, df, d, l, p0);
		if(f<0.0){
            		xl=rts;
        	} else {
            		xh=rts;
        	}
    	}

		std::cout << "ERROR (Bisection method)" << std::endl;
		throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
	}

SubVectorHandler& ModuleCatenary::AssRes(SubVectorHandler& WorkVec, doublereal dCoef, const VectorHandler& XCurr, const VectorHandler& XPrimeCurr)
	{
		integer iNumRows = 0;
		integer iNumCols = 0;	
		WorkSpaceDim(&iNumRows, &iNumCols);
		WorkVec.ResizeReset(iNumRows);

		integer iFirstMomIndex = g_pNode->iGetFirstMomentumIndex();
		for (int iCnt = 1; iCnt<=6; iCnt++) {
			WorkVec.PutRowIndex(iCnt, iFirstMomIndex + iCnt);
		}
		const Vec3 FP = g_pNode->GetXCurr();		
		GFP = Vec3(GFPx,GFPy,GFPz);
		AP  = Vec3(APx,APy,APz);

		FPx = FP.dGet(1);
		FPy = FP.dGet(2);
		FPz = FP.dGet(3);

		FP_AP  = FP  - AP;
		GFP_AP = GFP - AP;

		h = fabs(FP_AP.dGet(3));
	
		L0_APFP = L - h;
		L_APFP  = std::sqrt(std::pow(FP_AP.dGet(1), 2)+std::pow(FP_AP.dGet(2), 2));

		delta = L_APFP-L0_APFP;
		d	  = delta / h;
		l	  = L / h;
	
		if(d<=0) {
			H  = 0;
			V = w*h;
			p0 = 0;
		} else if(d>=(std::sqrt(std::pow(l,2)-1)-(l-1))) {
			std::cout << "ERROR (The length between anchor to fairlead  is over the length of chain)" << std::endl;
			throw ErrInterrupted(MBDYN_EXCEPT_ARGS);
		} else {
			x1 = 0;
			x2 = 1.e+6;
			double Ans_x = ModuleCatenary::rtsafe(x1, x2, xacc, d, l, p0);

			S = h*std::sqrt(1+2*Ans_x);
			H = Ans_x*w*h;
			V = w*S;
		}

		doublereal dFSF = FSF.dGet();
		H = H*dFSF;
		V = V*dFSF;

		if(FP_AP.dGet(1)>=0) {
			dFx = H*std::cos(std::atan((FP_AP.dGet(2))/(FP_AP.dGet(1))));
			dFy = H*std::sin(std::atan((FP_AP.dGet(2))/(FP_AP.dGet(1))));
			dFz = V;
		}
		else {
			dFx = H*std::cos(std::atan((FP_AP.dGet(2))/(FP_AP.dGet(1)))+M_PI);
			dFy = H*std::sin(std::atan((FP_AP.dGet(2))/(FP_AP.dGet(1)))+M_PI);
			dFz = V;
		}

		Fx = -dFx;
		Fy = -dFy;
		Fz = -dFz;

		F = Vec3(Fx,Fy,Fz);
		M = Vec3(0,0,0);

		WorkVec.Add(1,F);
		WorkVec.Add(4,M);

		return WorkVec;
	}

unsigned int ModuleCatenary::iGetNumPrivData(void) const
	{
		return 0;
	}

int ModuleCatenary::iGetNumConnectedNodes(void) const
	{
		return 0;
	}

void ModuleCatenary::GetConnectedNodes(std::vector<const Node *>& connectedNodes) const
	{
		connectedNodes.resize(0);
	}

void ModuleCatenary::SetValue(DataManager *pDM, VectorHandler& X, VectorHandler& XP, SimulationEntity::Hints *ph)
	{
		NO_OP;
	}

std::ostream& ModuleCatenary::Restart(std::ostream& out) const
	{
		return out << "# ModuleTemplate: not implemented" << std::endl;
	}

unsigned int ModuleCatenary::iGetInitialNumDof(void) const
	{
		return 0;
	}

void ModuleCatenary::InitialWorkSpaceDim(integer* piNumRows, integer* piNumCols) const
	{
		*piNumRows = 0;
		*piNumCols = 0;
	}

VariableSubMatrixHandler& ModuleCatenary::InitialAssJac( VariableSubMatrixHandler& WorkMat, const VectorHandler& XCurr)
	{
		ASSERT(0);

		WorkMat.SetNullMatrix();

		return WorkMat;
	}

SubVectorHandler& ModuleCatenary::InitialAssRes( SubVectorHandler& WorkVec, const VectorHandler& XCurr)
	{
		ASSERT(0);

		WorkVec.ResizeReset(0);

		return WorkVec;
	}

bool catenary_set(void) {
	#ifdef DEBUG
		std::cerr << __FILE__ <<":"<< __LINE__ << ":"<< __PRETTY_FUNCTION__ << std::endl;
	#endif

	UserDefinedElemRead *rf = new UDERead<ModuleCatenary>;

	if (!SetUDE("catenary", rf)) {
		delete rf;
		return false;
	}

	return true;
}

#ifndef STATIC_MODULES

extern "C" {

	int module_init(const char *module_name, void *pdm, void *php) {
		if (!catenary_set()) {
			silent_cerr("catenary: "
				"module_init(" << module_name << ") "
				"failed" << std::endl);
			return -1;
		}

		return 0;
	}

}
#endif
