wx = [2, 5, 7, 9, 11];

PCA_w_km = [0.2386, 0.2394, 0.2509, 0.2450, 0.2388];
ICA_w_km = [0.2532, 0.2658, 0.2487, 0.2707, 0.2860];
RP_w_km = [0.2068, 0.2476, 0.2319, 0.2119, 0.2174];

PCA_w_em = [0.2433, 0.3244, 0.3614, 0.3248, 0.3389];
ICA_w_em = [0.3115, 0.4171, 0.3654, 0.3329, 0.3456];
RP_w_em = [0.2680, 0.3244, 0.3250, 0.2974, 0.3344];

figure()
plot(wx, PCA_w_km, 'r-.', wx, ICA_w_km, 'g-.', wx, RP_w_km, 'b-.', ...
     wx, PCA_w_em, 'r-', wx, ICA_w_em, 'g-', wx, RP_w_em, 'b-');
axis([0 12 0 0.60])

lx = [2, 4, 6, 8, 10, 12, 14, 16];

PCA_l_km = [0.1508, 0.2229, 0.2619, 0.2826, 0.2834, 0.2767, 0.2938, 0.2906];
ICA_l_km = [0.1601, 0.2522, 0.3093, 0.3739, 0.3726, 0.4076, 0.4412, 0.4012];
RP_l_km = [0.1616, 0.2038, 0.2979, 0.2576, 0.2599, 0.2278, 0.2715, 0.3380];

PCA_l_em = [0.1599, 0.2862, 0.3412, 0.3802, 0.3888, 0.3675, 0.4007, 0.4073];
ICA_l_em = [0.1803, 0.3043, 0.3499, 0.3971, 0.4128, 0.4337, 0.4496, 0.4235];
RP_l_em = [0.1977, 0.2621, 0.3209, 0.3195, 0.3853, 0.3576, 0.3987, 0.4116];

figure()
plot(lx, PCA_l_km, 'r-.', lx, ICA_l_km, 'g-.', lx, RP_l_km, 'b-.', ...
     lx, PCA_l_em, 'r-', lx, ICA_l_em, 'g-', lx, RP_l_em, 'b-');
axis([0 12 0 0.60])