import pydicom as dicom 

path = "016_77.dcm"
rt_struct = "RS020001.dcm"

rt = dicom.read_file(rt_struct)

# the output is flattend vectors --> "x1,y1,z1,x2,y2,z2 .... xn,yn,zn"
# Z axis is related to the slice thickness -- The value could be imported according to the parameters of the mri image 
# more info --> https://dicom.innolitics.com/ciods/rt-structure-set/roi-contour/30060039/30060040/30060050
print(rt.ROIContourSequence[6].ContourSequence[15].ContourData)

# -14.249999523163