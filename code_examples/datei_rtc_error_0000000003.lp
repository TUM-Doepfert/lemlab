\* Source Pyomo model name=unknown *\

min 
objective_fun:
+0.5 deviation_gr_plus
+0.5 deviation_gr_minus
+0.4 deviation_bat_minus(BZ43TCmqtgBPhkT)
+0.4 deviation_bat_plus(BZ43TCmqtgBPhkT)
+1 deviation_ev_minus(7UOpIOGqpN3XK8L)
+1 deviation_ev_minus(BSMTXEJu1F2fkku)

s.t.

c_e_con_pv(HGJdmrAoFZmYLTD)_:
+1 p_pv(HGJdmrAoFZmYLTD)
= 0.0

c_l_con_hp_p(lDTA8TNIqBsmJBE)_:
+1 p_hp(lDTA8TNIqBsmJBE)
>= -7500

c_e_con_hp_q(lDTA8TNIqBsmJBE)_:
+3.03 p_hp(lDTA8TNIqBsmJBE)
+1 q_hp(lDTA8TNIqBsmJBE)
= 0

c_u_tes_soc_1(lDTA8TNIqBsmJBE)_:
-0.2777777777777778 q_tes_out(lDTA8TNIqBsmJBE)
+0.225 q_tes_in(lDTA8TNIqBsmJBE)
<= 48800.0

c_l_tes_soc_2(lDTA8TNIqBsmJBE)_:
-0.2777777777777778 q_tes_out(lDTA8TNIqBsmJBE)
+0.225 q_tes_in(lDTA8TNIqBsmJBE)
>= 0

c_u_tes_bin_minus(lDTA8TNIqBsmJBE)_:
+1 q_tes_in(lDTA8TNIqBsmJBE)
+100000 q_tes_milp(lDTA8TNIqBsmJBE)
<= 100000

c_u_tes_bin_plus(lDTA8TNIqBsmJBE)_:
+1 q_tes_out(lDTA8TNIqBsmJBE)
-100000 q_tes_milp(lDTA8TNIqBsmJBE)
<= 0

c_e_con_bat_dev(1)_:
-1 deviation_bat_minus(BZ43TCmqtgBPhkT)
+1 deviation_bat_plus(BZ43TCmqtgBPhkT)
+1 p_bat_out(BZ43TCmqtgBPhkT)
-1 p_bat_in(BZ43TCmqtgBPhkT)
= 0

c_u_bat_soc_1(BZ43TCmqtgBPhkT)_:
-0.2631578947368421 p_bat_out(BZ43TCmqtgBPhkT)
+0.2375 p_bat_in(BZ43TCmqtgBPhkT)
<= 10159.403941248347

c_l_bat_soc_2(BZ43TCmqtgBPhkT)_:
-0.2631578947368421 p_bat_out(BZ43TCmqtgBPhkT)
+0.2375 p_bat_in(BZ43TCmqtgBPhkT)
>= -640.5960587516527

c_u_bat_bin_minus(BZ43TCmqtgBPhkT)_:
+1 p_bat_in(BZ43TCmqtgBPhkT)
+100000 p_bat_milp(BZ43TCmqtgBPhkT)
<= 100000

c_u_bat_bin_plus(BZ43TCmqtgBPhkT)_:
+1 p_bat_out(BZ43TCmqtgBPhkT)
-100000 p_bat_milp(BZ43TCmqtgBPhkT)
<= 0

c_u_con_batt_charge_grid(1)_:
-1 p_pv(HGJdmrAoFZmYLTD)
+1 p_bat_in(BZ43TCmqtgBPhkT)
<= 0

c_u_con_ev_soc(1)_:
-0.25 p_ev_out(BSMTXEJu1F2fkku)
+0.25 p_ev_in(BSMTXEJu1F2fkku)
<= 34372.84343941984

c_l_con_ev_soc(2)_:
-0.25 p_ev_out(BSMTXEJu1F2fkku)
+0.25 p_ev_in(BSMTXEJu1F2fkku)
>= -25627.156560580162

c_u_con_ev_milp(1)_:
+1 p_ev_out(BSMTXEJu1F2fkku)
-1000000 p_ev_milp(BSMTXEJu1F2fkku)
<= 0

c_u_con_ev_milp(2)_:
+1 p_ev_in(BSMTXEJu1F2fkku)
+1000000 p_ev_milp(BSMTXEJu1F2fkku)
<= 1000000

c_u_con_ev_milp(3)_:
+1 deviation_ev_plus(BSMTXEJu1F2fkku)
-1000000 dev_ev_milp(BSMTXEJu1F2fkku)
<= 0

c_u_con_ev_milp(4)_:
+1 deviation_ev_minus(BSMTXEJu1F2fkku)
+1000000 dev_ev_milp(BSMTXEJu1F2fkku)
<= 1000000

c_e_con_ev_dev(1)_:
-1 deviation_ev_minus(BSMTXEJu1F2fkku)
+1 p_ev_out(BSMTXEJu1F2fkku)
-1 p_ev_in(BSMTXEJu1F2fkku)
+1 deviation_ev_plus(BSMTXEJu1F2fkku)
= 0

c_e_con_heat_balance_:
+1 q_hp(lDTA8TNIqBsmJBE)
+1 q_tes_out(lDTA8TNIqBsmJBE)
-1 q_tes_in(lDTA8TNIqBsmJBE)
= 27590.0

c_e_con_balance_:
+1 deviation_gr_plus
-1 deviation_gr_minus
+1 p_pv(HGJdmrAoFZmYLTD)
+1 p_hp(lDTA8TNIqBsmJBE)
+1 p_bat_out(BZ43TCmqtgBPhkT)
-1 p_bat_in(BZ43TCmqtgBPhkT)
+1 p_ev_out(BSMTXEJu1F2fkku)
-1 p_ev_in(BSMTXEJu1F2fkku)
-1 p_ev_in(7UOpIOGqpN3XK8L)
+1 p_ev_out(7UOpIOGqpN3XK8L)
= -6839.0

bounds
   0 <= deviation_gr_plus <= +inf
   0 <= deviation_gr_minus <= +inf
   0 <= deviation_bat_minus(BZ43TCmqtgBPhkT) <= +inf
   0 <= deviation_bat_plus(BZ43TCmqtgBPhkT) <= +inf
   0 <= deviation_ev_minus(7UOpIOGqpN3XK8L) <= +inf
   0 <= deviation_ev_minus(BSMTXEJu1F2fkku) <= +inf
   0 <= p_pv(HGJdmrAoFZmYLTD) <= +inf
   -inf <= p_hp(lDTA8TNIqBsmJBE) <= 0
   0 <= q_hp(lDTA8TNIqBsmJBE) <= +inf
   0 <= q_tes_out(lDTA8TNIqBsmJBE) <= 0
   0 <= q_tes_in(lDTA8TNIqBsmJBE) <= 195200
   0 <= q_tes_milp(lDTA8TNIqBsmJBE) <= 1
   0 <= p_bat_out(BZ43TCmqtgBPhkT) <= 10800
   0 <= p_bat_in(BZ43TCmqtgBPhkT) <= 10800
   0 <= p_bat_milp(BZ43TCmqtgBPhkT) <= 1
   0 <= p_ev_out(BSMTXEJu1F2fkku) <= 0
   0 <= p_ev_in(BSMTXEJu1F2fkku) <= 11000
   0 <= p_ev_milp(BSMTXEJu1F2fkku) <= 1
   0 <= deviation_ev_plus(BSMTXEJu1F2fkku) <= +inf
   0 <= dev_ev_milp(BSMTXEJu1F2fkku) <= 1
   0 <= p_ev_in(7UOpIOGqpN3XK8L) <= 0
   0 <= p_ev_out(7UOpIOGqpN3XK8L) <= 0
binary
  q_tes_milp(lDTA8TNIqBsmJBE)
  p_bat_milp(BZ43TCmqtgBPhkT)
  p_ev_milp(BSMTXEJu1F2fkku)
  dev_ev_milp(BSMTXEJu1F2fkku)
end
