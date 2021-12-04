import numpy as np
from astropy.io import fits
from scipy.ndimage.filters import convolve as convolveim
from matplotlib.pylab import mpl
from code_20190825 import ikernal23 as ikernel_svm


# 解决画图是中文显示的问题
mpl.rcParams['font.sans-serif'] = ['SimHei']

ker_size = 1
[F, Fr, Fc, Frr, Fcc, Frc] = ikernel_svm.get_par_der_2d_operator(dimension=2,ker_size=ker_size,sigam2=7,gama=100)
kernel = []
for item,item_name in zip([F, Fr, Fc, Frr, Fcc, Frc],['F', 'Fr', 'Fc', 'Frr', 'Fcc', 'Frc']):
    item_kernel = item.reshape(2 * ker_size +1, 2 * ker_size +1)
    kernel.append(item_kernel)
    np.savetxt(item_name + '.txt',item_kernel)

#%%

ker_size = 1
[F, Fr, Fc, Fh, Frr, Fcc, Fhh, Frc, Fch, Frh] = ikernel_svm.get_par_der_3d_operator(dimension=3,ker_size=ker_size,sigam2=7,gama=1)
kernel = []
for item in [F, Fr, Fc, Fh, Frr, Frc, Frh, Fcc, Fch, Fhh]:
    item_kernel = item.reshape(2 * ker_size +1, 2 * ker_size +1, 2 * ker_size +1)
    kernel.append(item_kernel)


L = kernel[0]
Lr = kernel[1]
Lc = kernel[2]
Lh = kernel[3]
Lrr = kernel[4]
Lrc = kernel[5]
Lrh = kernel[6]
Lcc = kernel[7]
Lch = kernel[8]
Lhh = kernel[9]


ker_size = 1
[F, Fr, Fc, Fh, Frr, Fcc, Fhh, Frc, Fch, Frh] = ikernel_svm.get_par_der_3d_operator(dimension=3,ker_size=ker_size,sigam2=7,gama=1)


kernel = []
for item in [Fr, Fc, Fh, Frr, Frc, Frh, Fcc, Fch, Fhh]:
    item_kernel = item.reshape(2 * ker_size +1, 2 * ker_size +1, 2 * ker_size +1)
    kernel.append(item_kernel)


# data = fits.getdata('data/single_clump/model_1.fits')
data = fits.getdata(r'data\simulated_clumps_medium\fits\model\s_model_000.fits')
print(data.shape)
print(data.max())
print(np.where(data==data.max()))

data_convv = {}
for i, item in enumerate(['dr', 'dc', 'dh','drr', 'drc', 'drh', 'dcc', 'dch', 'dhh']):
    data_convv[item] = convolveim(data, kernel[i], mode='constant')

Drr = data_convv['drr']
Drc = data_convv['drc']
Drh = data_convv['drh']
Dcc = data_convv['dcc']
Dch = data_convv['dch']
Dhh = data_convv['dhh']

Dr = data_convv['dr']
Dc = data_convv['dc']
Dh = data_convv['dh']


D1 = Drr
D2 = Drr * Dcc - Drc * Drc
D3 = Drr * (Dcc * Dhh - Dch * Dch) - Drc * (Drc * Dhh - Dch * Drh) + Drh * (Drc * Dch - Drh * Dcc)


index_D1 = D1 < 0
index_D2 = D2 > 0
index_D3 = D3 < 0

index_Dr = np.abs(Dr) < 5
index_Dc = np.abs(Dc) < 5
index_Dh = np.abs(Dh) < 5

index_Drch = index_Dr & index_Dc & index_Dh

# list(set(index_D1).intersection(set(index_D2)))
points = np.where((index_D1 & index_D2 & index_D3 & index_Drch)== True)
print(points)

#%%


local_points = []
for item in range(len(points[0])):
    x, y, z = points[0][item], points[1][item], points[2][item]
#     print([x,y,z,data[x,y,z]])
    local_points.append(data[x,y,z])
local_points = np.array(local_points)
print('*'*20)
print('检测得到的区域包含的点个数为：%d' %len(points[0]))

print('三维数据中的最大值为：%f' %data.max())
print('三维数据中的最大值d的下标为：%d %d %d' %np.where(data==data.max()))
print('*'*20)
print('检测区域中的最大值为：%f' %local_points.max())


index_all = (index_D1 & index_D2 & index_D3 & index_Drch)
print(type(index_all))
print(index_all.astype(np.float16).max())


noise = np.random.randn(100,100) * 1


from matplotlib import pyplot as plt
plt.hist(noise.reshape(1,10000),100)
