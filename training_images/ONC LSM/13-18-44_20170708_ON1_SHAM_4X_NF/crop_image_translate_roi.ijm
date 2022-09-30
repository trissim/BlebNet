makeRectangle(1161, 2047, 512, 512);
run("Duplicate...", "use");
count = roiManager("count");
array = newArray(count);
  for (i=0; i<array.length; i++) {
      array[i] = i;
  }
roiManager("select", array);
roiManager("translate", -1161, -2047);