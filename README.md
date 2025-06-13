# Bachelor's Thesis Informatiks
Everything in plotter.py reproduces and uses exactly what is in the original paper.
In thesis.py is everything (so far) that's new
determin_next_F(st) and determin_next_G(st) determine the next step deterministically based on the notes. pick_lyapunov(st, history, t) is determining it with the lyapunov drift method
calculate_next(st, dt) is returning either s(t+1) = 0 or s(t) + 1 based on the old s(t) as well as the next action to take (dt = 0,1,2) This is inheritly non-deterministic unless a specific seed is given for the random calculation.

Chanigng the debuggind_mode variable in line 16 from 0 (lyapunov) to 1 (G value in such a way that it will wait after each time step for user input, and display current AoSI) helps debugging. 
Switching between run_sim (calculate with F or G, has to be picked in the function) and run_sim_3 (calculate with Lyapunov drift) can be done in the main method relatively at the beginning.

To switch between functions F and G, just switch the F and G in determin_next_F/determin_next_G.

All constants are at the very top of the code and have descriptions next to them
