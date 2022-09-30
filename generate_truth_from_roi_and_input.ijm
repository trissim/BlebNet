input_dir = getDirectory("Choose an input images Directory");
roi_dir = getDirectory("Choose a ROI Directory");
save_dir = getDirectory("Choose a save Directory");
roi_list = getFileList(roi_dir);
input_list = getFileList(input_dir);

for (i = 0; i < roi_list.length; i++) {
	// delete everything in ROImanager
	for (index = 0; index < roiManager("count"); index++) {
		roiManager("delete");
		print(index);
	}
	roi_path = roi_dir + roi_list[i];
	input_path = input_dir + input_list[i];
	open(input_path);
	roiManager("Open", roi_path);
	run("Binary (0-255) mask(s) from Roi(s)", "save_mask(s) save_in=["+save_dir+"] suffix=_truth save_mask_as=tif rm=[RoiManager[size=15, visible=true]]");
	input_path;
  	close();
}
