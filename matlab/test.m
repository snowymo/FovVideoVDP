I_ref = imread("C:\Users\snowy\OneDrive\NewResults\fvvdp\foveation_sample\z=-70\gt.png");
I_gt = imread("C:\Users\snowy\OneDrive\NewResults\fvvdp\foveation_sample\z=-70\gt_RT_k3.0.png");
I_nerf = imread("C:\Users\snowy\OneDrive\NewResults\fvvdp\foveation_sample\z=-70\nerf_RT_k3.0.png");
I_our = imread("C:\Users\snowy\OneDrive\NewResults\fvvdp\foveation_sample\z=-70\our.png");

fixpoint = [757.55 780.73];
[blur_gt, diff_map_noise_gt] = fvvdp(I_gt, I_ref,'display_name', 'htc_vive_pro','foveated', true,'options', { 'fixation_point', fixpoint }, 'heatmap', 'threshold');
[blur_nerf, diff_map_noise_nerf] = fvvdp(I_nerf, I_ref,'display_name', 'htc_vive_pro','foveated', true,'options', { 'fixation_point', fixpoint }, 'heatmap', 'threshold');
[blur_our, diff_map_noise_our] = fvvdp(I_our, I_ref,'display_name', 'htc_vive_pro','foveated', true,'options', { 'fixation_point', fixpoint }, 'heatmap', 'threshold');

I_ref_kfr = imread("C:\Users\snowy\OneDrive\NewResults\fvvdp\foveation_sample\kfr\GT-kfr-1440.png");
I_gt_kfr = imread("C:\Users\snowy\OneDrive\NewResults\fvvdp\foveation_sample\kfr\F-GT-kfr1440.png");
I_nerf_beyond = imread("C:\Users\snowy\OneDrive\NewResults\fvvdp\foveation_sample\kfr\nerf-kfr1440.png");
I_our_kfr = imread("C:\Users\snowy\OneDrive\NewResults\fvvdp\foveation_sample\kfr\our-kfr1440.png");

[kfr_gt, diff_map_noise_gt_kfr] = fvvdp(I_gt_kfr, I_ref_kfr,'display_name', 'htc_vive_pro','foveated', true,'options', { 'fixation_point', fixpoint }, 'heatmap', 'threshold');
[kfr_nerf, diff_map_noise_nerf_kfr] = fvvdp(I_nerf_beyond, I_ref_kfr,'display_name', 'htc_vive_pro','foveated', true,'options', { 'fixation_point', fixpoint }, 'heatmap', 'threshold');
[kfr_our, diff_map_noise_our_kfr] = fvvdp(I_our_kfr, I_ref_kfr,'display_name', 'htc_vive_pro','foveated', true,'options', { 'fixation_point', fixpoint }, 'heatmap', 'threshold');

I_ref_beyond = imread("C:\Users\snowy\OneDrive\NewResults\fvvdp\foveation_sample\z=-70\gt.png");
I_gt_beyond = imread("C:\Users\snowy\OneDrive\NewResults\fvvdp\foveation_sample\z=-70\random\F-GT-hmd.png");
I_nerf_beyond = imread("C:\Users\snowy\OneDrive\NewResults\fvvdp\foveation_sample\z=-70\random\F-nerf-hmd.png");
I_our_beyond = imread("C:\Users\snowy\OneDrive\NewResults\fvvdp\foveation_sample\z=-70\our.png");

[beyond_gt, diff_map_noise_gt_beyond] = fvvdp(I_gt_beyond, I_ref_beyond,'display_name', 'htc_vive_pro','foveated', true,'options', { 'fixation_point', fixpoint }, 'heatmap', 'threshold');
[beyond_nerf, diff_map_noise_nerf_beyond] = fvvdp(I_nerf_beyond, I_ref_beyond,'display_name', 'htc_vive_pro','foveated', true,'options', { 'fixation_point', fixpoint }, 'heatmap', 'threshold');
[beyond_our, diff_map_noise_our_beyond] = fvvdp(I_our_beyond, I_ref_beyond,'display_name', 'htc_vive_pro','foveated', true,'options', { 'fixation_point', fixpoint }, 'heatmap', 'threshold');