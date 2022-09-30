import os
import tifffile
import numpy as np
import pandas as pd
from functional.data_functions_CLEANED import clean_edges, find_TP_FP_FN_from_seg


segmentation_image_folder_path = 'D:\\Tristan\\barbara_optic_nerves\\to_analyse\\crush_confocal\\analysis_results\\performance\\segmentations'
segmentation_image_folders = [os.path.join(segmentation_image_folder_path,folder) for folder in os.listdir(segmentation_image_folder_path)]    

truth_folder_path = 'D:\\Tristan\\barbara_optic_nerves\\to_analyse\\crush_confocal\\analysis_results\\performance\\truths'
truth_folders = [os.path.join(truth_folder_path,folder) for folder in os.listdir(truth_folder_path)]    


for segmentation_folder, truth_folder in zip(segmentation_image_folders,truth_folders):
    #foldername = os.path.basename(segmentation_folder)
    #sav_dir = os.path.join(os.path.dirname(os.path.dirname(segmentation_folder)),"analysis_results",foldername + '_analytic_results_z_'+str(min_slices))
    #all_results['nerve'].append(foldername)

    segmentation_images = [os.path.join(segmentation_folder,image) for image in os.listdir(segmentation_folder) if ".tif" in image]
    truth_images = [os.path.join(truth_folder,image) for image in os.listdir(truth_folder) if ".tif" in image]
    for segmentation_image,truth_image in zip(segmentation_images,truth_images):
        truth_image = tifffile.imread(truth_image)
        truth_image = np.expand_dims(truth_image,axis=2)
        segmentation_image = tifffile.imread(segmentation_image)
        segmentation_image = np.expand_dims(segmentation_image,axis=2)
        #truth_image_cleaned = clean_edges(truth_image, extra_z=1, extra_xy=3, skip_top=1)

        TP, FN, FP, truth_image, cleaned_seg = find_TP_FP_FN_from_seg(segmentation_image, truth_image, size_limit=2)
        if not FN == 0:
            print('not 0')
        if not FP == 0:
            print('not 0')
        if TP + FN == 0: TP;
        else: sensitivity = TP/(TP + FN);     # PPV

        if TP + FP == 0: TP;
        else: precision = TP/(TP + FP);     # precision



#df = pd.DataFrame(all_results)
#result_save_path = os.path.join(os.path.dirname(truth_folder_path),"analysis_results","human_counts.csv")
#df.to_csv(result_save_path)
#
#