from msilib.schema import File
import pydicom as dicom
import numpy as np
import cv2 as cv
import torch
from pydicom.uid import UID
from pydicom.dataset import FileDataset, FileMetaDataset
import tempfile

class RT_Struct():
    def __init__(self,rt_list):
        self.rt_list = rt_list
        self.rt_file = FileMetaDataset()

    
    def write_rt_metadata(self):
        self.rt_file.Modality = "RTSTRUCT"
        self.rt_file.InstitutionName = "CGFL"
        self.rt_file.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.481.3') # It is the same like the patients but with 481.3
        self.rt_file.MediaStorageSOPInstanceUID = UID('999.61977.8337.20211025142030867') # Check this id how is defined
        self.rt_file.TransferSyntaxUID
        self.rt_file.ImplementationClassUID
        self.rt_file.ImplementationVersionName
        self.rt_file.SpecificCharacterSet
        self.rt_file.SOPInstanceUID = self.rt_file.MediaStorageSOPInstanceUID
        self.rt_file.SOPClassUID = self.rt_file.MediaStorageSOPClassUID
        self.rt_file.StudyDate
        self.rt_file.SeriesDate
        self.rt_file.Manufacturer
        self.rt_file.InstitutionAddress
        self.rt_file.InstitutionalDepartmentName
        ManufacturerModelName
        PatientName
        PatientID
        PatientBirthDate
        PatientSex
        ClinicalTrialSponsorName
        ClinicalTrialProtocolID
        ClinicalTrialSubjectID
        StudyInstanceUID
        SeriesInstanceUID
        FrameOfReferenceUID
        StudyID




    def copy_patient_data(self,):
        # We need just one slice to copy the metadata
        rand_slice, _ = self.rt_list[2]
        self.write_rt_metadata(rand_slice)

