\* Source Pyomo model name=unknown *\

min 
objective_fun:
+0.5 deviation_gr_plus
+0.5 deviation_gr_minus
+0.4 deviation_bat_minus(dngLejqt7mnnqfQ)
+0.4 deviation_bat_plus(dngLejqt7mnnqfQ)
+1 deviation_ev_minus(fiQoFAafLrS1Hy5)
+1 deviation_ev_minus(qelyV7MjZuDE7Ww)

s.t.

c_e_con_pv(QC0QtBzR35tdDXk)_:
+1 p_pv(QC0QtBzR35tdDXk)
= 0.0

c_l_con_hp_p(P2ASy5Q04bp7yAG)_:
+1 p_hp(P2ASy5Q04bp7yAG)
>= -7600

c_e_con_hp_q(P2ASy5Q04bp7yAG)_:
+2.98 p_hp(P2ASy5Q04bp7yAG)
+1 q_hp(P2ASy5Q04bp7yAG)
= 0

c_u_tes_soc_1(P2ASy5Q04bp7yAG)_:
-0.2777777777777778 q_tes_out(P2ASy5Q04bp7yAG)
+0.225 q_tes_in(P2ASy5Q04bp7yAG)
<= 82930.17058333328

c_l_tes_soc_2(P2ASy5Q04bp7yAG)_:
-0.2777777777777778 q_tes_out(P2ASy5Q04bp7yAG)
+0.225 q_tes_in(P2ASy5Q04bp7yAG)
>= -57869.82941666672

c_u_tes_bin_minus(P2ASy5Q04bp7yAG)_:
+1 q_tes_in(P2ASy5Q04bp7yAG)
+100000 q_tes_milp(P2ASy5Q04bp7yAG)
<= 100000

c_u_tes_bin_plus(P2ASy5Q04bp7yAG)_:
+1 q_tes_out(P2ASy5Q04bp7yAG)
-100000 q_tes_milp(P2ASy5Q04bp7yAG)
<= 0

c_e_con_bat_dev(1)_:
-1 deviation_bat_minus(dngLejqt7mnnqfQ)
+1 deviation_bat_plus(dngLejqt7mnnqfQ)
+1 p_bat_out(dngLejqt7mnnqfQ)
-1 p_bat_in(dngLejqt7mnnqfQ)
= 0

c_u_bat_soc_1(dngLejqt7mnnqfQ)_:
-0.2631578947368421 p_bat_out(dngLejqt7mnnqfQ)
+0.2375 p_bat_in(dngLejqt7mnnqfQ)
<= 10200.0

c_l_bat_soc_2(dngLejqt7mnnqfQ)_:
-0.2631578947368421 p_bat_out(dngLejqt7mnnqfQ)
+0.2375 p_bat_in(dngLejqt7mnnqfQ)
>= 2.2737367544323206e-13

c_u_bat_bin_minus(dngLejqt7mnnqfQ)_:
+1 p_bat_in(dngLejqt7mnnqfQ)
+100000 p_bat_milp(dngLejqt7mnnqfQ)
<= 100000

c_u_bat_bin_plus(dngLejqt7mnnqfQ)_:
+1 p_bat_out(dngLejqt7mnnqfQ)
-100000 p_bat_milp(dngLejqt7mnnqfQ)
<= 0

c_u_con_batt_charge_grid(1)_:
-1 p_pv(QC0QtBzR35tdDXk)
+1 p_bat_in(dngLejqt7mnnqfQ)
<= 0

c_u_con_ev_soc(1)_:
-0.25 p_ev_out(fiQoFAafLrS1Hy5)
+0.25 p_ev_in(fiQoFAafLrS1Hy5)
<= 30000.0

c_l_con_ev_soc(2)_:
-0.25 p_ev_out(fiQoFAafLrS1Hy5)
+0.25 p_ev_in(fiQoFAafLrS1Hy5)
>= -30000.0

c_u_con_ev_soc(3)_:
-0.25 p_ev_out(qelyV7MjZuDE7Ww)
+0.25 p_ev_in(qelyV7MjZuDE7Ww)
<= 25485.117003367

c_l_con_ev_soc(4)_:
-0.25 p_ev_out(qelyV7MjZuDE7Ww)
+0.25 p_ev_in(qelyV7MjZuDE7Ww)
>= -34514.882996633

c_u_con_ev_milp(1)_:
+1 p_ev_out(fiQoFAafLrS1Hy5)
-1000000 p_ev_milp(fiQoFAafLrS1Hy5)
<= 0

c_u_con_ev_milp(2)_:
+1 p_ev_in(fiQoFAafLrS1Hy5)
+1000000 p_ev_milp(fiQoFAafLrS1Hy5)
<= 1000000

c_u_con_ev_milp(3)_:
+1 deviation_ev_plus(fiQoFAafLrS1Hy5)
-1000000 dev_ev_milp(fiQoFAafLrS1Hy5)
<= 0

c_u_con_ev_milp(4)_:
+1 deviation_ev_minus(fiQoFAafLrS1Hy5)
+1000000 dev_ev_milp(fiQoFAafLrS1Hy5)
<= 1000000

c_u_con_ev_milp(5)_:
+1 p_ev_out(qelyV7MjZuDE7Ww)
-1000000 p_ev_milp(qelyV7MjZuDE7Ww)
<= 0

c_u_con_ev_milp(6)_:
+1 p_ev_in(qelyV7MjZuDE7Ww)
+1000000 p_ev_milp(qelyV7MjZuDE7Ww)
<= 1000000

c_u_con_ev_milp(7)_:
+1 deviation_ev_plus(qelyV7MjZuDE7Ww)
-1000000 dev_ev_milp(qelyV7MjZuDE7Ww)
<= 0

c_u_con_ev_milp(8)_:
+1 deviation_ev_minus(qelyV7MjZuDE7Ww)
+1000000 dev_ev_milp(qelyV7MjZuDE7Ww)
<= 1000000

c_e_con_ev_dev(1)_:
-1 deviation_ev_minus(fiQoFAafLrS1Hy5)
+1 p_ev_out(fiQoFAafLrS1Hy5)
-1 p_ev_in(fiQoFAafLrS1Hy5)
+1 deviation_ev_plus(fiQoFAafLrS1Hy5)
= 0

c_e_con_ev_dev(2)_:
-1 deviation_ev_minus(qelyV7MjZuDE7Ww)
+1 p_ev_out(qelyV7MjZuDE7Ww)
-1 p_ev_in(qelyV7MjZuDE7Ww)
+1 deviation_ev_plus(qelyV7MjZuDE7Ww)
= 0

c_e_con_heat_balance_:
+1 q_hp(P2ASy5Q04bp7yAG)
+1 q_tes_out(P2ASy5Q04bp7yAG)
-1 q_tes_in(P2ASy5Q04bp7yAG)
= 36733.0

c_e_con_balance_:
+1 deviation_gr_plus
-1 deviation_gr_minus
+1 p_pv(QC0QtBzR35tdDXk)
+1 p_hp(P2ASy5Q04bp7yAG)
+1 p_bat_out(dngLejqt7mnnqfQ)
-1 p_bat_in(dngLejqt7mnnqfQ)
+1 p_ev_out(fiQoFAafLrS1Hy5)
-1 p_ev_in(fiQoFAafLrS1Hy5)
+1 p_ev_out(qelyV7MjZuDE7Ww)
-1 p_ev_in(qelyV7MjZuDE7Ww)
= -2595.0

bounds
   0 <= deviation_gr_plus <= +inf
   0 <= deviation_gr_minus <= +inf
   0 <= deviation_bat_minus(dngLejqt7mnnqfQ) <= +inf
   0 <= deviation_bat_plus(dngLejqt7mnnqfQ) <= +inf
   0 <= deviation_ev_minus(fiQoFAafLrS1Hy5) <= +inf
   0 <= deviation_ev_minus(qelyV7MjZuDE7Ww) <= +inf
   0 <= p_pv(QC0QtBzR35tdDXk) <= +inf
   -inf <= p_hp(P2ASy5Q04bp7yAG) <= 0
   0 <= q_hp(P2ASy5Q04bp7yAG) <= +inf
   0 <= q_tes_out(P2ASy5Q04bp7yAG) <= 7600.0
   0 <= q_tes_in(P2ASy5Q04bp7yAG) <= 7600.0
   0 <= q_tes_milp(P2ASy5Q04bp7yAG) <= 1
   0 <= p_bat_out(dngLejqt7mnnqfQ) <= 10200
   0 <= p_bat_in(dngLejqt7mnnqfQ) <= 10200
   0 <= p_bat_milp(dngLejqt7mnnqfQ) <= 1
   0 <= p_ev_out(fiQoFAafLrS1Hy5) <= 0
   0 <= p_ev_in(fiQoFAafLrS1Hy5) <= 11000
   0 <= p_ev_out(qelyV7MjZuDE7Ww) <= 0
   0 <= p_ev_in(qelyV7MjZuDE7Ww) <= 11000
   0 <= p_ev_milp(fiQoFAafLrS1Hy5) <= 1
   0 <= deviation_ev_plus(fiQoFAafLrS1Hy5) <= +inf
   0 <= dev_ev_milp(fiQoFAafLrS1Hy5) <= 1
   0 <= p_ev_milp(qelyV7MjZuDE7Ww) <= 1
   0 <= deviation_ev_plus(qelyV7MjZuDE7Ww) <= +inf
   0 <= dev_ev_milp(qelyV7MjZuDE7Ww) <= 1
binary
  q_tes_milp(P2ASy5Q04bp7yAG)
  p_bat_milp(dngLejqt7mnnqfQ)
  p_ev_milp(fiQoFAafLrS1Hy5)
  dev_ev_milp(fiQoFAafLrS1Hy5)
  p_ev_milp(qelyV7MjZuDE7Ww)
  dev_ev_milp(qelyV7MjZuDE7Ww)
end
