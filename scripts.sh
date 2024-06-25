
# Train LeRF-Linear Model
# python resample/train_model.py -e models/lerf-l/ --twoStage --outC 1 --linear 

# Train LeRF-Gaussian Model
python resample/train_model.py -e models/lerf-g/ --twoStage --outC 3

# Train LeRF-Net Model
# python resample/train_model.py -e models/lerf-net --model IMDN2 --twoStage  --featC 3 --inC 3

# train LeRF-Net++ model
# data/DIV2KPreUpsample/
#             /HR/*.png
#             /LR_bicubic/X2/*.png # Place X2 pre-upsampled results here
# python resample/train_model.py --trainDir data/DIV2KPreUpsample --valDir data/rrPreUpsample --valWDir data/WarpPreUpsample --scale 2 --model IMDN2 --featC 3 --inC 3 -e models/lerf-net-plus --twoStage

# Transfer network to LUTs for LeRF-L and LeRF-G
# python resample/transfer_to_lut.py -e models/lerf-l/ --outC 1
# Resulting LUT size:  (83521, 1, 1, 1) Saved to models/lerf-l/LUT_s2_sr0.npy
# ...
python resample/transfer_to_lut.py -e models/lerf-g/ --outC 3
# Resulting LUT size:  (83521, 3, 1, 1) Saved to models/lerf-g/LUT_s2_sr0.npy
# ...
# Resulting LUT size:  (83521, 1, 1, 1) Saved to models/lerf-g/LUT_s1_sr0.npy
# ...

# Fine-tune LeRF-Linear LUT
# python resample/train_model.py -e models/lerf-l/ --lutft --model SWF2LUT  --twoStage --outC 1 --batchSize 256 --linear
# Fine-tune LeRF-Gaussian LUT
python resample/train_model.py -e models/lerf-g/ --lutft --model SWF2LUT --twoStage --outC 3 --batchSize 256

# Evaulation for arbitrary-scale upsampling
# python resample/eval_lut_sr.py --testDir data/rrBenchmark --resultRoot results/sr --lutName LUTft -e models/lerf-l --linear
# Scale           2.0x2.0         3.0x3.0         4.0x4.0
# Set5            34.84/0.9432    30.72/0.8773    29.13/0.8270
python resample/eval_lut_sr.py --testDir data/rrBenchmark --resultRoot results/sr --lutName LUTft -e models/lerf-g
# Scale           2.0x2.0         3.0x3.0         4.0x4.0
# Set5            35.71/0.9475    32.02/0.8980    30.15/0.8548


# Evaulation for homographic warping
# python resample/eval_lut_warp.py --testDir data/WarpBenchmark --resultRoot results/warp --lutName LUTft -e models/lerf-l --linear
# Scale           isc             osc
# Set5            32.90   27.13
python resample/eval_lut_warp.py --testDir data/WarpBenchmark --resultRoot results/warp --lutName LUTft -e models/lerf-g
# Scale           isc             osc
# Set5            33.81   27.89

# Evaulation of LeRF-Net and LeRF-Net++
# python resample/eval_model.py --testDir data/rrBenchmark --resultRoot results/sr -e models/lerf_net --model IMDN2 --twoStage  --featC 3 --inC 3
# Scale           2.0x2.0         3.0x3.0         4.0x4.0
# Set5            36.57/0.9562    33.04/0.9162    31.26/0.8810
# python resample/eval_model.py --testDir data/WarpBenchmark --resultRoot results/warp -e models/lerf_net --model IMDN2 --twoStage  --featC 3 --inC 3
# Scale           isc             osc
# Set5            34.72   28.61

# datasets/rrPreUpsample/
#                    /[testset]/HR/*.png
#                              /LR_bicubic/X2/*.png
#                              /LR_bicubic/X1.50_2.00/*.png # Place X2 pre-upsampled results here
#                    /...
# python resample/eval_model.py --testDir data/rrPreUpsample --resultRoot results/sr -e models/lerf_net_plus --model IMDN2 --twoStage --featC 3 --inC 3
# Scale           2.0x2.0         3.0x3.0         4.0x4.0
# Set5            38.27/0.9616    34.57/0.9289    32.52/0.8983

# datasets/WarpPreUpsample/
#                    /[testset]/HR/*.png
#                              /isc/*.png # Place X2 pre-upsampled results here
#                              /isc/*.pth
#                              /osc/*.png
#                              /osc/*.pth
#                    /...
# python resample/eval_model.py --testDir data/WarpPreUpsample --resultRoot results/warp -e models/lerf_net_plus --model IMDN2 --twoStage --featC 3 --inC 3
# Scale           isc             osc
# Set5            35.78   28.89


##########################
# Notes on Warping mPSNR #
##########################
# In previous works like LTEW, mPSNR results (Table 3 in the LTEW paper) of SR benchmark datasets (e.g., Set5) is reported after converting to a gray channel.
# (Please refer to https://github.com/jaewon-lee-b/ltew/blob/706affb6680efca8eba9f17dd8ddb42995478be0/utils.py#L165)
# Here, we maintain the same metric function as DIV2K-Warping, where mPSNR is measured by averaging PSNR across RGB channels, i.e., cPSNR.
# PSNR on Y (gray) channel is usually much higher than the three-channel cPSNR or mPSNR. 
# In our work, we only report mPSNR on DIV2K-Warping to avoid inconsistency (Table IV in the extended LeRF paper).
