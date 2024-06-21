\* Source Pyomo model name=unknown *\

min 
objective_fun:
+0.5 deviation_gr_plus
+0.5 deviation_gr_minus
+0.4 deviation_bat_minus(yKl44QKRIlLEHGJ)
+0.4 deviation_bat_plus(yKl44QKRIlLEHGJ)
+1 deviation_ev_minus(ksNCtqqxTyP0O8U)

s.t.

c_u_con_pv(XuAjGwOIptQlnKq)_:
+1 p_pv(XuAjGwOIptQlnKq)
<= 0.0

c_u_con_pv(x7YcP8XjyRs3WAd)_:
+1 p_pv(x7YcP8XjyRs3WAd)
<= 0.0

c_l_con_hp_p(845CQgIhpdTAXGO)_:
+1 p_hp(845CQgIhpdTAXGO)
>= -14000

c_e_con_hp_q(845CQgIhpdTAXGO)_:
+3.73 p_hp(845CQgIhpdTAXGO)
+1 q_hp(845CQgIhpdTAXGO)
= 0

c_u_tes_soc_1(845CQgIhpdTAXGO)_:
-0.2777777777777778 q_tes_out(845CQgIhpdTAXGO)
+0.225 q_tes_in(845CQgIhpdTAXGO)
<= 263200.0

c_l_tes_soc_2(845CQgIhpdTAXGO)_:
-0.2777777777777778 q_tes_out(845CQgIhpdTAXGO)
+0.225 q_tes_in(845CQgIhpdTAXGO)
>= 1.2732925824820995e-11

c_u_tes_bin_minus(845CQgIhpdTAXGO)_:
+1 q_tes_in(845CQgIhpdTAXGO)
+100000 q_tes_milp(845CQgIhpdTAXGO)
<= 100000

c_u_tes_bin_plus(845CQgIhpdTAXGO)_:
+1 q_tes_out(845CQgIhpdTAXGO)
-100000 q_tes_milp(845CQgIhpdTAXGO)
<= 0

c_e_con_bat_dev(1)_:
-1 deviation_bat_minus(yKl44QKRIlLEHGJ)
+1 deviation_bat_plus(yKl44QKRIlLEHGJ)
+1 p_bat_out(yKl44QKRIlLEHGJ)
-1 p_bat_in(yKl44QKRIlLEHGJ)
= 1976.0

c_u_bat_soc_1(yKl44QKRIlLEHGJ)_:
-0.2631578947368421 p_bat_out(yKl44QKRIlLEHGJ)
+0.2375 p_bat_in(yKl44QKRIlLEHGJ)
<= 22706.30789987475

c_l_bat_soc_2(yKl44QKRIlLEHGJ)_:
-0.2631578947368421 p_bat_out(yKl44QKRIlLEHGJ)
+0.2375 p_bat_in(yKl44QKRIlLEHGJ)
>= -1193.6921001252488

c_u_bat_bin_minus(yKl44QKRIlLEHGJ)_:
+1 p_bat_in(yKl44QKRIlLEHGJ)
+100000 p_bat_milp(yKl44QKRIlLEHGJ)
<= 100000

c_u_bat_bin_plus(yKl44QKRIlLEHGJ)_:
+1 p_bat_out(yKl44QKRIlLEHGJ)
-100000 p_bat_milp(yKl44QKRIlLEHGJ)
<= 0

c_u_con_batt_charge_grid(1)_:
-1 p_pv(XuAjGwOIptQlnKq)
-1 p_pv(x7YcP8XjyRs3WAd)
+1 p_bat_in(yKl44QKRIlLEHGJ)
<= 0

c_u_con_ev_soc(1)_:
-0.25 p_ev_out(ksNCtqqxTyP0O8U)
+0.25 p_ev_in(ksNCtqqxTyP0O8U)
<= 0.0

c_l_con_ev_soc(2)_:
-0.25 p_ev_out(ksNCtqqxTyP0O8U)
+0.25 p_ev_in(ksNCtqqxTyP0O8U)
>= -60000.0

c_u_con_ev_milp(1)_:
+1 p_ev_out(ksNCtqqxTyP0O8U)
-1000000 p_ev_milp(ksNCtqqxTyP0O8U)
<= 0

c_u_con_ev_milp(2)_:
+1 p_ev_in(ksNCtqqxTyP0O8U)
+1000000 p_ev_milp(ksNCtqqxTyP0O8U)
<= 1000000

c_u_con_ev_milp(3)_:
+1 deviation_ev_plus(ksNCtqqxTyP0O8U)
-1000000 dev_ev_milp(ksNCtqqxTyP0O8U)
<= 0

c_u_con_ev_milp(4)_:
+1 deviation_ev_minus(ksNCtqqxTyP0O8U)
+1000000 dev_ev_milp(ksNCtqqxTyP0O8U)
<= 1000000

c_e_con_ev_dev(1)_:
-1 deviation_ev_minus(ksNCtqqxTyP0O8U)
+1 p_ev_out(ksNCtqqxTyP0O8U)
-1 p_ev_in(ksNCtqqxTyP0O8U)
+1 deviation_ev_plus(ksNCtqqxTyP0O8U)
= 0

c_e_con_heat_balance_:
+1 q_hp(845CQgIhpdTAXGO)
+1 q_tes_out(845CQgIhpdTAXGO)
-1 q_tes_in(845CQgIhpdTAXGO)
= 52790.0

c_e_con_balance_:
+1 deviation_gr_plus
-1 deviation_gr_minus
+1 p_pv(XuAjGwOIptQlnKq)
+1 p_pv(x7YcP8XjyRs3WAd)
+1 p_hp(845CQgIhpdTAXGO)
+1 p_bat_out(yKl44QKRIlLEHGJ)
-1 p_bat_in(yKl44QKRIlLEHGJ)
+1 p_ev_out(ksNCtqqxTyP0O8U)
-1 p_ev_in(ksNCtqqxTyP0O8U)
= -317.0

bounds
   0 <= deviation_gr_plus <= +inf
   0 <= deviation_gr_minus <= +inf
   0 <= deviation_bat_minus(yKl44QKRIlLEHGJ) <= +inf
   0 <= deviation_bat_plus(yKl44QKRIlLEHGJ) <= +inf
   0 <= deviation_ev_minus(ksNCtqqxTyP0O8U) <= +inf
   0 <= p_pv(XuAjGwOIptQlnKq) <= +inf
   0 <= p_pv(x7YcP8XjyRs3WAd) <= +inf
   -inf <= p_hp(845CQgIhpdTAXGO) <= 0
   0 <= q_hp(845CQgIhpdTAXGO) <= +inf
   0 <= q_tes_out(845CQgIhpdTAXGO) <= 420000
   0 <= q_tes_in(845CQgIhpdTAXGO) <= 70000
   0 <= q_tes_milp(845CQgIhpdTAXGO) <= 1
   0 <= p_bat_out(yKl44QKRIlLEHGJ) <= 23900
   0 <= p_bat_in(yKl44QKRIlLEHGJ) <= 23900
   0 <= p_bat_milp(yKl44QKRIlLEHGJ) <= 1
   0 <= p_ev_out(ksNCtqqxTyP0O8U) <= 0
   0 <= p_ev_in(ksNCtqqxTyP0O8U) <= 11000
   0 <= p_ev_milp(ksNCtqqxTyP0O8U) <= 1
   0 <= deviation_ev_plus(ksNCtqqxTyP0O8U) <= +inf
   0 <= dev_ev_milp(ksNCtqqxTyP0O8U) <= 1
binary
  q_tes_milp(845CQgIhpdTAXGO)
  p_bat_milp(yKl44QKRIlLEHGJ)
  p_ev_milp(ksNCtqqxTyP0O8U)
  dev_ev_milp(ksNCtqqxTyP0O8U)
end
