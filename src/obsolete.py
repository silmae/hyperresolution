"""A file for obsolete code: things not likely to be needed, but ones I'm too scared to remove altogether"""


################ Ceres Occator images, unusable due to overexposure in the faculae ################

training_data = nn.TrainingData(type='DAWN_ISIS', filepath=Path('./datasets/DAWN/ISIS/m-VIR_IR_1B_1_494253260_1.cub')) # Ceres, HAMO, Occator
training_data = nn.TrainingData(type='DAWN_ISIS', filepath=Path('./datasets/DAWN/ISIS/m-VIR_VIS_1B_1_493567338_1.cub')) # Ceres, HAMO, Occator

VIR_IR_1B_1_494253260_1 = {'rot_deg': 64,
                           'crop_indices_x': (150, 475),
                           'crop_indices_y': (130, 380)}
VIR_VIS_1B_1_494253260_1 = {'rot_deg': 64,
                            'crop_indices_x': (154, 479),
                            'crop_indices_y': (130, 380)}

# For the dictionary in constants
'm-VIR_IR_1B_1_494253260_1.cub': VIR_IR_1B_1_494253260_1,  # Ceres, HAMO, Occator
'm-VIR_VIS_1B_1_494253260_1.cub': VIR_VIS_1B_1_494253260_1,


################ Loading files or file types that are not used anymore ################

# isisimage = CubeFile.open("./datasets/DAWN/ISIS/m-VIR_IR_1B_1_589173531_1.cub")  # This is the largest image of these, Ceres
# isisimage = CubeFile.open("./datasets/DAWN/ISIS/m-VIR_IR_1B_1_487367451_1.cub")  # A bit smaller image, Ceres
# isisimage = CubeFile.open("./datasets/DAWN/ISIS/f-VIR_IR_1B_1_483703316_1.cub")  # This is taken from afar and has half of Ceres in view
# isisimage_IR = CubeFile.open("./datasets/DAWN/ISIS/m-VIR_IR_1B_1_494387713_1.cub")
# isisimage_VIS = CubeFile.open("./datasets/DAWN/ISIS/m-VIR_VIS_1B_1_494387713_1.cub")

training_data = nn.TrainingData(type='remote_sensing', filepath=Path('./datasets/TinyAPEX.mat'))  # Remote sensing datacube used by Palsson et al. in the original paper
training_data = nn.TrainingData(type='rock', filepath=Path('./datasets/0065/A.mhdr.h5'))
training_data = nn.TrainingData(type='luigi', filepath=Path('./datasets/Luigi_stone/30klx_G2.nc'))
training_data = nn.TrainingData(type='DAWN_PDS3', filepath=Path('./datasets/DAWN/PDS3/VIR_VIS_1B_1_487349955_1.LBL'))

# For TrainingData initializer in nn.py
if type == 'remote_sensing':
    h, w, l, abundance_count, cube = file_handling.file_loader_rem_sens(filepath)
elif type == 'rock':
    h, w, l, cube, wavelengths = file_handling.file_loader_rock(filepath)
elif type == 'luigi':
    h, w, l, cube, wavelengths = file_handling.file_loader_luigi(filepath)

if type != 'remote_sensing':  # no wavelength data for the remote sensing images used here
    self.wavelengths = wavelengths
    # self.abundance_count = abundance_count

# For loading the files
def file_loader_rem_sens(filepath="./datasets/TinyAPEX.mat"):
    mat = scipy.io.loadmat(filepath)
    # mat = scipy.io.loadmat("./datasets/Samson.mat")
    w = mat['W'][0][0]  # W is 2 dim matrix with 1 element
    h = mat['H'][0][0]
    l = mat['L'][0][0]
    abundance_count = mat['p'][0][0]
    cube_flat = np.array(mat['Y'])
    cube_flat = cube_flat.transpose()
    # plt.imshow(cube_flat)
    # plt.show()
    cube = cube_flat.reshape(h, w, l)

    return h, w, l, abundance_count, cube


def file_loader_rock(filepath):
    d = h5py.File(filepath, 'r')
    contents = list(d.keys())
    cube = np.transpose(d['/hdr'], (1, 2, 0))
    cube = np.nan_to_num(cube, nan=1)  # Original rock images are masked with NaN, replace those with a number

    wavelengths = d['/wavelengths'][:]

    # cube = cube[50:, 50:, :]

    # # Sanity check plot
    # plt.imshow(np.nanmean(cube, 2))
    # plt.show()

    dimensions = cube.shape
    h = dimensions[0]
    w = dimensions[1]
    l = dimensions[2]

    return h, w, l, cube, wavelengths


def file_loader_luigi(filepath):
    """
    Loads data captured at University of Jyväskylä with an imager nicknamed "Luigi", saved as netcdf4 files.
    :param filepath:
        Path to file
    :return: h, w, l, cube, wavelengths:
        Height, width, length, image cube as ndarray, wavelength vector

    """
    data = xr.open_dataset(filepath)['reflectance']
    cube = data.values

    # The image is very large, crop to get manageable training time
    cube = cube[75:300, 100:300, 0:120]

    # # Sanity check plot
    # plt.imshow(cube[:, :, 80])
    # plt.show()

    shape = cube.shape
    w = shape[1]
    h = shape[0]
    l = shape[2]

    wavelengths = data.wavelength.values

    return h, w, l, cube, wavelengths


################ Stuff related to loss function ################

# https://discuss.pytorch.org/t/custom-loss-functions/29387 -Kimmo

# loss_short = MAPE(short_y_true, short_y_pred)

# This thing can kill backprop even when called for a single spectrum
# metric_SAM = torchmetrics.SpectralAngleMapper(reduction='none').to(device)
# loss_short_SAM = metric_SAM(short_y_pred, short_y_true)  # Using this function for masked cube breaks backprop: "RuntimeError: Function 'CatBackward0' returned nan values in its 0th output."
# loss_long_SAM = metric_SAM(long_y_pred, long_y_true)

# # Calculate long wavelength loss by comparing mean spectra of long wavelength cubes
# long_y_true = torch.mean(long_y_true, dim=(
#     2, 3))  # Moved this calculation to preprocessing and only feed the mean spectrum into here
# long_y_true = torch.unsqueeze(long_y_true, 2)
# long_y_true = torch.unsqueeze(long_y_true, 3)

# # TV over output spectra
# total_variation = torch.norm(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :], p=2)

# # Some custom metrics
# def MAPE(Y_actual, Y_Predicted):
#     abs_diff = torch.abs((Y_actual - Y_Predicted))
#     # zero mask
#     mask = (Y_actual != 0)
#     # initialize output tensor with desired value
#     norm = torch.full_like(Y_actual, fill_value=float('nan'))
#     norm[mask] = torch.div(abs_diff[mask], Y_actual[mask]) # divide by non-zero elements
#     norm = torch.nan_to_num(norm, posinf=1e10) # replace infinities with finite big number
#     mean_diff = torch.mean(norm)
#     mape = mean_diff * 100
#     return mape
#
# def mean_spectral_gradient(cube):
#     abs_grad = torch.abs(cube[:, 1:, :, :] - cube[:, :-1, :, :])
#     sum_grad = torch.sum(abs_grad, dim=(1))
#     mean_grad = torch.mean(sum_grad)
#     # sum_grad = torch.sum(mean_grad)
#     return mean_grad


################ Failed attempt to add TV reg to loss after calculating it in loss_fn ################

# # FOR SOME REASON THE TV TERM BACKPROPAGATION DOES NOT WORK
            # # Total variation regularization of endmember spectra
            # TV = torch.tensor(0.0, requires_grad=True)
            # # TV = torch.autograd.Variable(TV, requires_grad=True)
            # TV = TV.to(device)
            # metric_tv = torchmetrics.image.TotalVariation().to(device)
            #
            # for i in range(common_params['endmember_count']):
            #     kernel = dec.layers[-1].weight.data[:, i, :, :]
            #     # TV = TV + torch.norm(kernel[1:, :, :] - kernel[:-1, :, :], p=2)
            #     TV = TV + metric_tv(torch.unsqueeze(kernel, dim=0))
            #
            # TV_lambda = 1
            # loss = loss + TV_lambda * TV


################ Yeah just use Path from pathlib instead ################
def join(*args) -> str:
    """Custom join function to avoid problems using os.path.join. """

    n = len(args)
    s = ''
    for i,arg in enumerate(args):
        if i == n-1:
            s = s + arg
        else:
            s = s + arg + '/'
    p = os.path.normpath(s)
    return p


################ TODO Make this work someday ################

# def ASPECTify(cube, wavelengths, FWHMs, VIS=False, NIR1=True, NIR2=True, SWIR=True):
#     """Take a spectral image and make it look like data from Milani's ASPECT"""
#     if VIS:
#         print('Sorry, the function can not currently work with the VIS portion of ASPECT')
#         print('Stopping execution')
#         exit(1)
#
#     # # ASPECT wavelength vectors: change these values later, if the wavelengths change!
#     # ASPECT_VIS_wavelengths = np.linspace(start=0.650, stop=0.950, num=14)
#     # ASPECT_NIR1_wavelengths = np.linspace(start=0.850, stop=0.1250, num=14)
#     # ASPECT_NIR2_wavelengths = np.linspace(start=1.200, stop=0.1600, num=14)
#     # ASPECT_SWIR_wavelengths = np.linspace(start=1.650, stop=2.500, num=30)
#     # # ASPECT FOVs in degrees
#     # ASPECT_VIS_FOV = 10  # 10x10 deg square
#     # ASPECT_NIR_FOV_w = 6.7  # width
#     # ASPECT_NIR_FOV_h = 5.4  # height
#     # ASPECT_SWIR_FOV = 5.85  # circular
#
#     # Resample spectra to resemble ASPECT data
#     cube, wavelengths, FWHMs = ASPECT_resampling(cube, wavelengths, FWHMs)
#
#     if (NIR1 or NIR2) and SWIR:
#         # The largest is the SWIR circular FOV, and so it is the limiting factor
#         # Cut the largest possible rectangle where one side is circular FOV, other is width of NIR. Then divide
#         # along wavelength into NIR and SWIR, mask SWIR into circle and take mean spectrum, and cut NIR to size.
#         return None


################ The rest is abandoned code saved for snippets if needed ################

# def predict(r_m, t_m, nn_name='nn_default'):
#     """Use neural network to predict HyperBlend leaf model parameters from measured reflectance and transmittance.
#
#     :param r_m:
#         Measured reflectance.
#     :param t_m:
#         Measured transmittance.
#     :param nn_name:
#         Neural network name. Default name 'nn_default' is used if not given.
#         Provide only if you want to use your trained custom NN.
#     :return:
#         Lists ad, sd, ai, mf (absorption density, scattering desnity, scattering anisotropy, and mixing factor).
#         Use ``leaf_commons._convert_raw_params_to_renderable()`` before passing them to rendering method.
#     """
#
#     net = _load_model(nn_name=nn_name)
#     r_m = np.array(r_m)
#     t_m = np.array(t_m)
#     res = net(from_numpy(np.column_stack([r_m, t_m])))
#     res_item = res.detach().numpy()
#     ad = np.clip(res_item[:,0], 0., 1.)
#     sd = np.clip(res_item[:,1], 0., 1.)
#     ai = np.clip(res_item[:,2], 0., 1.)
#     mf = np.clip(res_item[:,3], 0., 1.)
#     return ad, sd, ai, mf


# def _load_model(nn_name):
#     """Loads the NN from disk.
#
#     :param nn_name:
#         Name of the model file.
#     :return:
#         Returns loaded NN.
#     :exception:
#         ModuleNotFoundError can happen if the network was trained when the name
#         of this script was something else than what it is now. Your only help
#         is to train again.
#     """
#     try:
#         p = _get_model_path(nn_name)
#         net = load(p)
#         net.eval()
#         logging.info(f"NN model loaded from '{p}'")
#     except ModuleNotFoundError as e:
#         logging.error(f"Pytorch could not load requested neural network. "
#                       f"This happens if class or file names associated with "
#                       f"NN are changed. You must train a new model to fix this.")
#         raise
#     return net


# def _get_model_path(nn_name='nn_default'):
#     """Returns path to the NN model.
#
#     :param nn_name:
#         Name of the NN.
#     :return:
#         Returns path to the NN model.
#     :exception:
#         FileNotFoundError if the model cannot be found.
#     """
#
#     if not nn_name.endswith('.pt'):
#         nn_name = nn_name + '.pt'
#     model_path = PH.join(PH.path_directory_surface_model(), nn_name)
#     if os.path.exists(model_path):
#         return model_path
#     else:
#         raise FileNotFoundError(f"Model '{model_path}' was not found. Check spelling.")


# def exists(nn_name='nn_default.pt'):
#     """Checks whether NN with given name exists.
#
#     :return:
#         True if found, False otherwise.
#     """
#
#     if not nn_name.endswith('.pt'):
#         nn_name = nn_name + '.pt'
#     model_path = PH.join(PH.path_directory_surface_model(), nn_name)
#     return os.path.exists(model_path)



