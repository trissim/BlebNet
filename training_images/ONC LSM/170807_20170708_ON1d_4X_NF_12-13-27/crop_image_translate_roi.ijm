makeRectangle(273, 991, 511, 511);
run("Duplicate...", "use");
count = roiManager("count");
array = newArray(count);
  for (i=0; i<array.length; i++) {
      array[i] = i;
  }
roiManager("select", array);
roiManager("translate", -273, -991);