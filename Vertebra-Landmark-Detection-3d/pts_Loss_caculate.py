import numpy as np
pts7_upsample    = [[29, 120, 188], [38, 172, 156], [38, 172, 224], [86, 176, 160], [86, 176, 212], [85, 116, 184], [130, 192, 160], [130, 192, 204], [133, 132, 180], [174, 208, 156], [174, 204, 200], [177, 148, 180], [218, 220, 152], [218, 216, 200], [222, 164, 176]]
pts7_upsample_gt = [[24, 120, 188], [32, 172, 156], [36, 172, 224], [84, 176, 160], [84, 176, 212], [88, 120, 184], [128, 192, 160], [128, 192, 208], [136, 136, 180], [172, 208, 156], [172, 208, 204], [180, 152, 180], [212, 216, 152], [216, 216, 200], [220, 164, 172]]
pts8_upsample    = [[25, 132, 192], [34, 192, 136], [34, 196, 248], [81, 124, 192], [78, 200, 148], [82, 204, 236], [126, 216, 152], [126, 216, 232], [129, 136, 192], [170, 232, 156], [174, 236, 224], [178, 152, 192], [218, 248, 156], [219, 252, 220], [222, 168, 188]]
pts8_upsample_gt = [[28, 132, 192], [36, 196, 140], [36, 200, 248], [80, 124, 192], [80, 200, 148], [80, 204, 236], [124, 216, 156], [124, 216, 232], [128, 136, 196], [168, 232, 156], [168, 236, 228], [176, 156, 188], [216, 248, 156], [216, 252, 220], [220, 168, 188]]
voxel_spacing = np.asarray([0.8,0.546,0.546],dtype=np.float32)

pts7_upsample_gt = np.asarray(pts7_upsample_gt,dtype=np.float32)
pts7_upsample = np.asarray(pts7_upsample,dtype=np.float32)

pts8_upsample_gt = np.asarray(pts8_upsample_gt,dtype=np.float32)
pts8_upsample = np.asarray(pts8_upsample,dtype=np.float32)

gap_7 = ((pts7_upsample_gt - pts7_upsample)) * voxel_spacing
gap_8 = ((pts8_upsample_gt - pts8_upsample)) * voxel_spacing

loss_7 = 0
loss_8 = 0

MSE_loss = 0
print('MSE_loss :',MSE_loss)

for i in range(15):
    loss_7 += np.sum(np.square(gap_7[i]))
    loss_8 += np.sum(np.square(gap_8[i]))

loss_7 /= 15
loss_8 /= 15

print(loss_7,loss_8)
print(abs(gap_7[:-1]).sum()/45,abs(gap_8[:-1]).sum()/45)












