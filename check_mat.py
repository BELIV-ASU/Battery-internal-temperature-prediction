import scipy.io as sio

path = r'C:\Users\yuvav\battery_data\Temperature_10C\DV_WLTP\59294.mat'
mat  = sio.loadmat(path)

print('Keys in file:')
for key in mat.keys():
    if not key.startswith('__'):
        val = mat[key]
        print('  ' + str(key) + ' -- shape: ' + str(val.shape) + ' -- dtype: ' + str(val.dtype))
        if val.size < 10:
            print('    values: ' + str(val.flatten()))