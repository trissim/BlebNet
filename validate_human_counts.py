import os
import tifffile
import numpy as np
import pandas as pd
from functional.data_functions_3D import process_output

min_slices = 2
min_slices_list=[min_slices]
all_results = {v:[] for v in ["nerve"] + [str(slices) for slices in min_slices_list] }
input_image_folder_path = 'D:\\Tristan\\barbara_optic_nerves\\to_analyse\\crush_confocal\\analyze'
input_image_folders = [os.path.join(input_image_folder_path,folder) for folder in os.listdir(input_image_folder_path)]    

human_segmentation_folder_path = 'D:\\Tristan\\barbara_optic_nerves\\to_analyse\\crush_confocal\\human_segmentations'
human_segmentation_folders = [os.path.join(human_segmentation_folder_path,folder) for folder in os.listdir(human_segmentation_folder_path)]    


for input_folder, human_segmentation_folder in zip(input_image_folders,human_segmentation_folders):
    foldername = os.path.basename(input_folder)
    sav_dir = os.path.join(os.path.dirname(os.path.dirname(input_folder)),"analysis_results",foldername + '_analytic_results_z_'+str(min_slices))
    all_results['nerve'].append(foldername)
    input_im_stack = [];
    output_stack = [];
    input_images = [os.path.join(input_folder,image) for image in os.listdir(input_folder) if ".tif" in image]
    human_segmentations = [os.path.join(human_segmentation_folder,image) for image in os.listdir(human_segmentation_folder) if ".tif" in image]
    for input_image,segmentation_image in zip(input_images,human_segmentations):

        filename = os.path.basename(input_image)
        filename = filename.split('.')[0:-1]
        filename = '.'.join(filename)
        segmentation_image = tifffile.imread(segmentation_image)
        input_image = tifffile.imread(input_image)
        """ Save as 3D stack """
        if len(output_stack) == 0:
            output_stack = segmentation_image
            #output_stack_masked = seg_train_masked
            input_im_stack = input_image
        else:
            """ Keep this if want to keep all the outputs as single stack """
            output_stack = np.dstack([output_stack, segmentation_image])
            input_im_stack = np.dstack([input_im_stack, input_image])

    all_results = process_output(output_stack,input_im_stack,min_slices,all_results,filename,sav_dir)


df = pd.DataFrame(all_results)
result_save_path = os.path.join(os.path.dirname(human_segmentation_folder_path),"analysis_results","human_counts.csv")
df.to_csv(result_save_path)