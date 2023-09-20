import numpy as np
from pathlib import Path

VIR_IR_1B_1_494387713_1 = {'rot_deg': -19,
                           'crop_indices_x': (130, 580),
                           'crop_indices_y': (190, 460)}
VIR_VIS_1B_1_494387713_1 = {'rot_deg': -19,
                            'crop_indices_x': (132, 582),
                            'crop_indices_y': (193, 463)}

VIR_IR_1B_1_486828195_1 = {'rot_deg': -20,
                           'crop_indices_x': (170, 460),
                           'crop_indices_y': (115, 270)}
VIR_VIS_1B_1_486828195_1 = {'rot_deg': -20,
                            'crop_indices_x': (171, 461),
                            'crop_indices_y': (116, 271)}

VIR_IR_1B_1_366636556_1 = {'rot_deg': -20,
                           'crop_indices_x': (101, 351),
                           'crop_indices_y': (104, 190)}
VIR_VIS_1B_1_366636556_1 = {'rot_deg': -20,
                            'crop_indices_x': (100, 350),
                            'crop_indices_y': (106, 192)}

VIR_IR_1B_1_367917915_1 = {'rot_deg': 52,
                           'crop_indices_x': (90, 250),
                           'crop_indices_y': (216, 354)}
VIR_VIS_1B_1_367917915_1 = {'rot_deg': 52,
                            'crop_indices_x': (93, 253),
                            'crop_indices_y': (214, 352)}

edge_detector_params = [50, 50]  # not exactly sure what these do: fiddle with them until you see a nice amount of edges
edge_detection_channel = 100  # index of wavelength channel used for edge detection: change this if you see no edges

Dawn_ISIS_rot_deg_and_crop_indices = {'m-VIR_IR_1B_1_494387713_1.cub': VIR_IR_1B_1_494387713_1,
                                      'm-VIR_VIS_1B_1_494387713_1.cub': VIR_VIS_1B_1_494387713_1,
                                      'm-VIR_IR_1B_1_486828195_1.cub': VIR_IR_1B_1_486828195_1,
                                      'm-VIR_VIS_1B_1_486828195_1.cub': VIR_VIS_1B_1_486828195_1,
                                      'm-VIR_IR_1B_1_366636556_1.cub': VIR_IR_1B_1_366636556_1,
                                      'm-VIR_VIS_1B_1_366636556_1.cub': VIR_VIS_1B_1_366636556_1,
                                      'm-VIR_IR_1B_1_367917915_1.cub': VIR_IR_1B_1_367917915_1,
                                      'm-VIR_VIS_1B_1_367917915_1.cub': VIR_VIS_1B_1_367917915_1
                                      }

ASPECT_wavelengths = np.asarray(np.linspace(start=0.850, stop=2.500, num=60))
ASPECT_FWHMs = np.zeros(shape=ASPECT_wavelengths.shape) + 0.040
ASPECT_NIR_channel_shape = (512, 640)

ceres_hc_dist = (2.55 + 2.99) / 2  # average between perihelion and aphelion

solar_path = Path('./datasets/E490_solar_spectrum_0.2_to_6micron.csv')
