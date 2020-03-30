import os
import argparse

import nibabel as nb

from .peer_func import *

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', help="Output dir to save models", required=True)
parser.add_argument('--stimulus_path', help="CSV with stimuli timing info", required=True)
parser.add_argument('--eye_mask_path', help="Eye mask Nifti", required=True)
parser.add_argument('--train_file', help="Training functional data (calibration)")
parser.add_argument('--test_file', help="Test functional data (regression)")
parser.add_argument('--use_gsr', help="Use global signal regression", action="store_true")
parser.add_argument('--use_ms', help="Use motion scrubbing", action="store_true")
parser.add_argument('--motion_fd_train', help=".1D file with framewise displacement of train func")
parser.add_argument('--motion_threshold', help="Threshold for FD")

args = parser.parse_args()

print('\nLoad Data')
print('====================================================')

eye_mask_path = args.eye_mask_path
eye_mask = nb.load(eye_mask_path).get_data()



if args.train_file:

    data = load_data(args['train_file'])
    for vol in range(data.shape[3]):
        output = np.multiply(eye_mask, data[:, :, :, vol])
        data[:, :, :, vol] = output
    volumes = data.shape[3]

    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                vmean = np.mean(np.array(data[x, y, z, :]))
                vstdv = np.std(np.array(data[x, y, z, :]))
                for time in range(volumes):
                    if vstdv != 0:
                        data[x, y, z, time] = (float(data[x, y, z, time]) - float(vmean)) / vstdv
                    else:
                        data[x, y, z, time] = float(data[x, y, z, time]) - float(vmean)

    if args.use_gsr:
        print('\nGlobal Signal Regression')
        print('====================================================')
        data = global_signal_regression(data, eye_mask_path)

    if args.use_ms:
        thresh = args.motion_threshold
        print(str('\nMotion Scrubbing').format(thresh))
        print('====================================================')

        ms_filename = args.motion_fd_train
        removed_indices = motion_scrub(ms_filename, '', thresh)
    else:
        removed_indices = None

    processed_data, calibration_points_removed = prepare_data_for_svr(data, removed_indices, eye_mask_path)

    print('\nTrain PEER')
    print('====================================================')

    xmodel, ymodel = train_model(processed_data, calibration_points_removed, args.stimulus_path)
    save_model(xmodel, ymodel, args.train_file, args.use_ms, args.use_gsr, args.output_dir)

    print('\nTraining done')



if args.test_file:

    filepath = args.test_file

    print('\nLoad Test Data')
    print('====================================================')

    data = load_data(filepath)
    for vol in range(data.shape[3]):
        output = np.multiply(eye_mask, data[:, :, :, vol])
        data[:, :, :, vol] = output
    volumes = data.shape[3]

    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                vmean = np.mean(np.array(data[x, y, z, :]))
                vstdv = np.std(np.array(data[x, y, z, :]))
                for time in range(volumes):
                    if vstdv != 0:
                        data[x, y, z, time] = (float(data[x, y, z, time]) - float(vmean)) / vstdv
                    else:
                        data[x, y, z, time] = float(data[x, y, z, time]) - float(vmean)

    if args.use_gsr:
        print('\nGlobal Signal Regression')
        print('====================================================')
        data = global_signal_regression(data, eye_mask_path)

    xmodel, ymodel, xmodel_name, ymodel_name = load_model(args.output_dir)


    print('\nPredicting Fixations')
    print('====================================================')

    raveled_data = [data[:, :, :, vol].ravel() for vol in np.arange(data.shape[3])]
    x_fix, y_fix = predict_fixations(xmodel, ymodel, raveled_data)
    x_fixname, y_fixname = save_fixations(x_fix, y_fix, xmodel_name, ymodel_name, args.output_dir)

    print('Fixations saved to specified output directory.')


    print('\nEstimating Eye Movements')
    print('====================================================')

    estimate_em(x_fix, y_fix, x_fixname, y_fixname, args.output_dir)

    print('Eye movements saved to specified output directory.')