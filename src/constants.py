import numpy as np

VIR_IR_1B_1_494387713_1 = {'rot_deg': -19,
                           'crop_indices_x': (130, 580),
                           'crop_indices_y': (190, 460)}
VIR_VIS_1B_1_494387713_1 = {'rot_deg': -19,
                            'crop_indices_x': (132, 582),
                            'crop_indices_y': (193, 463)}

Dawn_ISIS_rot_deg_and_crop_indices = {'m-VIR_IR_1B_1_494387713_1.cub': VIR_IR_1B_1_494387713_1,
                                      'm-VIR_VIS_1B_1_494387713_1.cub': VIR_VIS_1B_1_494387713_1}

ASPECT_wavelengths = np.asarray(np.linspace(start=0.850, stop=2.500, num=60))
ASPECT_FWHMs = np.zeros(shape=ASPECT_wavelengths.shape) + 0.040

