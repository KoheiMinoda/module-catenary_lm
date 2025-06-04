import math
import numpy as np


def myasinh(X):
    return math.log(X + math.sqrt(X*X + 1.0))


def myacosh(X):
    if X < 1.0:
        if X > 1.0 - 1e-9:
            X = 1.0
        else:
            raise ValueError(f"[Error] Domain error for myacosh: X = {X} < 1")
    return math.log(X + math.sqrt(X*X - 1.0))


class CatenarySolver:
    
    def __init__(self, fp_coords, ap_coords, l_total, xacc=1e-4):

        self.fp_coords = fp_coords
        self.ap_coords = ap_coords
        self.l_total = l_total
        self.xacc = xacc
        
        self.MAXIT_P0_SOLVER = 100
        self.MAXIT_RTSAFE = 200
        
        self.h_span = abs(fp_coords["z"] - ap_coords["z"])
        self.l_apfp = math.sqrt((fp_coords["x"] - ap_coords["x"])**2 +
                               (fp_coords["y"] - ap_coords["y"])**2)
        
        self.d_param = (self.l_apfp - (l_total - self.h_span)) / self.h_span
        self.l_param = l_total / self.h_span
        
        self._current_p0_for_funcd = 0.0
        
    def _solve_for_p0_in_funcd(self, x_candidate, l_val, xacc_p0):
        
        p0 = 0.0

        for _ in range(self.MAXIT_P0_SOLVER):
            cos_p0 = math.cos(p0)
            if abs(cos_p0) < 1e-12:
                break 

            tan_p0 = math.tan(p0)
            func1 = 1.0 / x_candidate + 1.0 / cos_p0
            
            sqrt_arg_f1 = func1**2 - 1.0
            
            if sqrt_arg_f1 < 0 and sqrt_arg_f1 > -1e-9: 
                sqrt_arg_f1 = 0.0
            elif sqrt_arg_f1 < 0:
                return p0

            sqrt_val_f1 = math.sqrt(sqrt_arg_f1)
            f1_for_p0 = x_candidate * (sqrt_val_f1 - tan_p0) - l_val

            if abs(f1_for_p0) < xacc_p0:
                return p0

            if abs(sqrt_val_f1) < 1e-9: 
                break

            term_A_df1 = func1 * tan_p0 / (cos_p0 * sqrt_val_f1)
            term_B_df1 = 1.0 / (cos_p0**2) 
            df1_for_p0 = x_candidate * (term_A_df1 - term_B_df1)

            if abs(df1_for_p0) < 1e-12:
                break
            
            p0 = p0 - f1_for_p0 / df1_for_p0
            
            if not (-math.pi/2 * 0.99 < p0 < math.pi/2 * 0.99):
                break
                
        return p0

    def _funcd_equations(self, x_val, d_val, l_val, xacc_for_p0_solver):

        f_res, df_res = 0.0, 0.0
        p0_local = 0.0

        if abs(x_val) < 1e-9:
            f_res = -d_val
            df_res = 0.0 
            p0_local = 0.0

        elif x_val > 0:
            if l_val <= 0: 
                X1 = 1.0 / x_val + 1.0
                term_acosh_X1 = myacosh(X1) 
                sqrt_1_2x = math.sqrt(1.0 + 2.0 * x_val)
                f_res = x_val * term_acosh_X1 - sqrt_1_2x + 1.0 - d_val
                
                sqrt_X1sq_m1 = math.sqrt(max(0, X1**2 - 1.0))
                if abs(x_val * sqrt_X1sq_m1) < 1e-9: 
                    df_res = term_acosh_X1 - (1.0 / sqrt_1_2x) 
                else:
                    df_res = term_acosh_X1 - (1.0 / sqrt_1_2x) - (1.0 / (x_val * sqrt_X1sq_m1))
                
                p0_local = 0.0

            else: 
                threshold_x = (l_val**2 - 1.0) / 2.0

                if x_val > threshold_x:
                    p0_local = self._solve_for_p0_in_funcd(x_val, l_val, xacc_for_p0_solver)
                    tan_p0 = math.tan(p0_local)
                    X2 = l_val / x_val + tan_p0
                    asinh_X2 = myasinh(X2)
                    asinh_tan_p0 = myasinh(tan_p0)
                    f_res = x_val * (asinh_X2 - asinh_tan_p0) - l_val + 1.0 - d_val
                    
                    sqrt_X2sq_p1 = math.sqrt(X2**2 + 1.0)
                    if abs(x_val * sqrt_X2sq_p1) < 1e-9:
                        df_res = asinh_X2 - asinh_tan_p0
                    else:
                        df_res = (asinh_X2 - asinh_tan_p0) - l_val / (x_val * sqrt_X2sq_p1)
                
                else: 
                    X5 = 1.0 / x_val + 1.0
                    term_acosh_X5 = myacosh(X5)
                    sqrt_1_2x_alt = math.sqrt(1.0 + 2.0 * x_val)
                    f_res = x_val * term_acosh_X5 - sqrt_1_2x_alt + 1.0 - d_val

                    sqrt_X5sq_m1 = math.sqrt(max(0, X5**2 - 1.0))
                    if abs(x_val * sqrt_X5sq_m1) < 1e-9:
                        df_res = term_acosh_X5 - (1.0 / sqrt_1_2x_alt)
                    else:
                        df_res = term_acosh_X5 - (1.0 / sqrt_1_2x_alt) - (1.0 / (x_val * sqrt_X5sq_m1))
                    p0_local = 0.0

        else:
            raise ValueError("[Error] x_val < 0 encountered in _funcd_equations")
        
        self._current_p0_for_funcd = p0_local
        return f_res, df_res, p0_local

    def _rtsafe_solver(self, x1_rt, x2_rt, xacc_rt, d_rt, l_rt, xacc_p0_rt):

        p0_for_root = 0.0 
        
        f_low, _, p0_low = self._funcd_equations(x1_rt, d_rt, l_rt, xacc_p0_rt)
        f_high, _, p0_high = self._funcd_equations(x2_rt, d_rt, l_rt, xacc_p0_rt)

        if (f_low > 0 and f_high > 0) or (f_low < 0 and f_high < 0):
            raise ValueError(f"[Error] Root not bracketed in rtsafe: f({x1_rt})={f_low}, f({x2_rt})={f_high}")

        if f_low == 0: 
            p0_for_root = p0_low
            return x1_rt, p0_for_root
        if f_high == 0: 
            p0_for_root = p0_high
            return x2_rt, p0_for_root

        if f_low < 0:
            xl_rt, xh_rt = x1_rt, x2_rt
        else:
            xl_rt, xh_rt = x2_rt, x1_rt
        
        rts_curr = 0.5 * (x1_rt + x2_rt)
        dx_old = abs(x2_rt - x1_rt)
        dx_curr = dx_old
        
        f_curr, df_curr, p0_curr = self._funcd_equations(rts_curr, d_rt, l_rt, xacc_p0_rt)
        p0_for_root = p0_curr

        for _ in range(self.MAXIT_RTSAFE):
            if (((rts_curr - xh_rt) * df_curr - f_curr) * ((rts_curr - xl_rt) * df_curr - f_curr) > 0.0) or \
               (abs(2.0 * f_curr) > abs(dx_old * df_curr)):
                dx_old = dx_curr
                dx_curr = 0.5 * (xh_rt - xl_rt)
                rts_curr = xl_rt + dx_curr

                if xl_rt == rts_curr:
                    _, _, p0_for_root = self._funcd_equations(rts_curr, d_rt, l_rt, xacc_p0_rt)
                    return rts_curr, p0_for_root
            
            else:
                dx_old = dx_curr
                if abs(df_curr) < 1e-12:
                    dx_curr = 0.5 * (xh_rt - xl_rt)
                    rts_curr = xl_rt + dx_curr

                    if xl_rt == rts_curr: 
                         _, _, p0_for_root = self._funcd_equations(rts_curr, d_rt, l_rt, xacc_p0_rt)
                         return rts_curr, p0_for_root
                
                else:
                    dx_curr = f_curr / df_curr

                temp_rts = rts_curr
                rts_curr -= dx_curr
                if temp_rts == rts_curr: 
                    _, _, p0_for_root = self._funcd_equations(rts_curr, d_rt, l_rt, xacc_p0_rt)
                    return rts_curr, p0_for_root
            
            if abs(dx_curr) < xacc_rt:
                _, _, p0_for_root = self._funcd_equations(rts_curr, d_rt, l_rt, xacc_p0_rt)
                return rts_curr, p0_for_root

            f_curr, df_curr, p0_curr = self._funcd_equations(rts_curr, d_rt, l_rt, xacc_p0_rt)
            p0_for_root = p0_curr 

            if f_curr < 0:
                xl_rt = rts_curr
            else:
                xh_rt = rts_curr
                
        raise RuntimeError("[Error] Max iterations exceeded in rtsafe")

    def solve_catenary_parameter(self):

        if self.d_param <= 0:
            return 0.0
        
        elif self.l_param >= 1.0 and self.d_param >= (math.sqrt(self.l_param**2 - 1.0) - (self.l_param - 1.0)):
            raise ValueError("[Error] Line length L_total is too short")

        else:
            rt_x1 = self.xacc
            rt_x2 = max(100.0, self.l_param * self.l_param * 2.0) 

            f1_check, _, _ = self._funcd_equations(rt_x1, self.d_param, self.l_param, self.xacc)
            f2_check, _, _ = self._funcd_equations(rt_x2, self.d_param, self.l_param, self.xacc)

            if (f1_check * f2_check) > 0:
                rt_x1 = self.xacc
                rt_x2 = 1.0e6 
                f1_check_wide, _, _ = self._funcd_equations(rt_x1, self.d_param, self.l_param, self.xacc)
                f2_check_wide, _, _ = self._funcd_equations(rt_x2, self.d_param, self.l_param, self.xacc)
                if (f1_check_wide * f2_check_wide) > 0:
                     raise ValueError("[Error] Cannot bracket root for rtsafe even with wide range")
                else:
                     print(f"Using wide bracket for rtsafe: [{rt_x1}, {rt_x2}]")
            
            ans_x_final, _ = self._rtsafe_solver(rt_x1, rt_x2, self.xacc, self.d_param, self.l_param, self.xacc)
            return ans_x_final

    def generate_initial_nodes(self, num_internal_nodes=19):
 
        ans_x_final = self.solve_catenary_parameter()
        a_catenary = ans_x_final * self.h_span
        
        print(f"Calculated Ans_x: {ans_x_final:.6f}")
        print(f"Catenary parameter 'a': {a_catenary:.3f} m")
        
        xf, zf = self.fp_coords["x"], self.fp_coords["z"]
        xa, za = self.ap_coords["x"], self.ap_coords["z"]
        
        Xs_rel = xa - xf 
        Zs_rel = za - zf
        S_line = self.l_total

        if abs(a_catenary) < 1e-6:
            if abs(self.l_apfp) < 1e-6:
                nodes = []
                num_segments = num_internal_nodes + 1
                s_seg_vert = self.l_total / num_segments
                z_start_hang = max(zf, za) 
                x_hang = xf 
                y_hang = 0.0
                
                for i_node in range(1, num_internal_nodes + 1):
                    s_node_from_top = i_node * s_seg_vert
                    node_z_val = zf - s_node_from_top 
                    
                    nodes.append({
                        "id": i_node, 
                        "x": x_hang, 
                        "y": y_hang, 
                        "z": node_z_val, 
                        "s_from_fp": s_node_from_top
                    })
                
                return nodes, a_catenary
            else: 
                print("Ans_x=0 with L_APFP > 0. Catenary failed.")
                return [], a_catenary

        else:
            denominator_term_sum_x = 2 * math.sinh(Xs_rel / (2 * a_catenary))
            
            if abs(denominator_term_sum_x) < 1e-9: 
                sum_norm_x_half = Zs_rel / Xs_rel if abs(Xs_rel) > 1e-6 else 0.0 
            else:
                sum_norm_x_half = myasinh((Zs_rel / a_catenary) / denominator_term_sum_x)

            diff_norm_x_half = Xs_rel / (2 * a_catenary)
            
            x1_n = sum_norm_x_half - diff_norm_x_half 
            x2_n = sum_norm_x_half + diff_norm_x_half 

            x_gm = xf - a_catenary * x1_n
            z_offset = zf - a_catenary * math.cosh(x1_n)

            print(f"Global catenary: a={a_catenary:.3f}, x_gm={x_gm:.3f}, z_offset={z_offset:.3f}")
            
            num_segments = num_internal_nodes + 1
            s_segment_len = S_line / num_segments 

            internal_nodes_coords = []

            for i_node in range(1, num_internal_nodes + 1):
                s_k_arc_from_fp = i_node * s_segment_len 

                arg_for_asinh = s_k_arc_from_fp / a_catenary + math.sinh(x1_n)
                xk_n = myasinh(arg_for_asinh)
                
                node_x_coord = x_gm + a_catenary * xk_n
                node_y_coord = 0.0 
                node_z_coord = a_catenary * math.cosh(xk_n) + z_offset
                
                internal_nodes_coords.append({
                    "id": i_node,
                    "x": node_x_coord,
                    "y": node_y_coord,
                    "z": node_z_coord,
                    "s_from_fp": s_k_arc_from_fp
                })
            
            print("----------------------------------------------------------------------")
            print("Anchor Point:   X={:.3f}, Y={:.3f}, Z={:.3f}".format(
              self.ap_coords['x'], self.ap_coords['y'], self.ap_coords['z']))
            
            return internal_nodes_coords, a_catenary
