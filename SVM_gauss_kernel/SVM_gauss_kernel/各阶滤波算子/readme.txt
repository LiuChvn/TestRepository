ker_size = 1;
[F, Fr, Fc, Frr, Fcc, Frc] = ikernel_svm.get_par_der_2d_operator(dimension=2,ker_size=ker_size,sigam2=7,gama=100)
kernel = []
for item,item_name in zip([F, Fr, Fc, Frr, Fcc, Frc],['F', 'Fr', 'Fc', 'Frr', 'Fcc', 'Frc']):
    item_kernel = item.reshape(2 * ker_size +1, 2 * ker_size +1)
    kernel.append(item_kernel)
    np.savetxt(item_name + '.txt',item_kernel)