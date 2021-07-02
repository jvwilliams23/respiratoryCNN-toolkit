import getopt
import sys
import os
from os import path 
import stat
import numpy as np
import SimpleITK as sitk
from skimage.measure import label, regionprops #, regionprops_table
import math
import matplotlib.pyplot as plt
import scipy.signal
import pandas
islobe=False
debug=False
airways_to_lobes=False
def myshow(img, zpos="default", title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    if zpos=="default":
        zpos=nda.shape[0] // 2
    print(zpos)

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if c not in (3, 4):
            nda = nda[zpos, :, :]

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if c not in (3, 4):
            raise RuntimeError("Unable to show 3D-vector Image")

        # take a z-slice
        nda = nda[zpos, :, :, :]

    xsize = nda.shape[1]
    ysize = nda.shape[0]

    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * xsize / dpi, (1 + margin) * ysize / dpi

    plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
    ax = plt.gca()

    extent = (0, xsize * spacing[0], ysize * spacing[1], 0)

    t = ax.imshow(nda, extent=extent, interpolation=None)

    if nda.ndim == 2:
        t.set_cmap("gray")

    if(title):
        plt.title(title)

    plt.show()

def get_hist(image):
    result = sitk.GetArrayFromImage(image)
    print(type(result))
    print(result.shape)
    plt.figure('histogram')
    result = result.flatten()
    n, bins, patches = plt.hist(result, bins=256, range= (1,result.max()),normed=0, facecolor='red', alpha=0.75,histtype = 'step')
    plt.show()

def main(argv):
    inputdirseg=False
    outputdirseg=False
    inputdirscan=False
    ouputdirscan=False
    inputdirectoryseg = ''
    outputdirectoryseg = ''
    inputdirectoryscan=''
    outputdirectoryscan=''
    try:
        opts, args = getopt.getopt(argv,"ha:b:c:d:l:e:",["inputdirectoryseg=","outputdirectoryseg=", "inputdirectoryscan=","outputdirectoryscan=","=islobe","=airways_to_lobes"])
    except getopt.GetoptError as e:
        print(e)
        print ("crop_inputdata.py -a <inputdirectoryseg> -b <outputdirectoryseg> -c <inputdirectoryscan> -d <outputdirectoryscan> -l <islobe> -e <airways_to_lobes")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ("crop_inputdata.py -a <inputdirectoryseg> -b <outputdirectoryseg> -c <inputdirectoryscan> -d <outputdirectoryscan> -l <islobe> -e <airways_to_lobes>")
            sys.exit(2)
        elif opt in ("-a", "--isegdirectory"):
            inputdirectoryseg = arg
            inputdirseg=True
        elif opt in ("-b", "--osegdirectory"):
            outputdirectoryseg = arg
            outputdirseg=True
        elif opt in ("-c", "--iscandirectory"):
            inputdirectoryscan = arg
            inputdirscan=True
        elif opt in ("-d", "--oscandirectory"):
            outputdirectoryscan = arg
            outputdirscan=True
        elif opt in("-l","--islobe"):
            global islobe
            islobe=True
        elif opt in ("-e","--airways_to_lobes"):
            global airways_to_lobes
            airways_to_lobes=True

        else:
            assert False, "unhandled option"
    if inputdirseg==False or inputdirscan==False:
        sys.exit(2)
        print("oops need an input directory, use -h for help")
    else:
         check_dir(inputdirectoryseg)
         check_dir(inputdirectoryscan)
    if outputdirseg==False:
        outputdirectoryseg=os.getcwd()
        #make a directory to output to
        access_rights=0o777
        outputdirectoryseg=outputdirectoryseg+os.sep+"cropped_scans"
        try:
            os.mkdir(str(outputdirseg),access_rights)
        except OSError:
            if (os.exists(outputdirectoryseg+os.sep+"cropped_scans")):
                pass
    if outputdirscan==False:
        outputdirectoryscan=os.getcwd()
        #make a directory to output to
        access_rights=0o777
        outputdirectoryscan=outputdirectoryscan+os.sep+"cropped_scans"
        try:
            os.mkdir(str(outputdirscan),access_rights)
        except OSError:
            if (os.exists(outputdirectoryscan+os.sep+"cropped_scans")):
                pass
    else: 
        check_dir(outputdirectoryscan)
        check_dir(outputdirectoryseg)

    return inputdirectoryseg,outputdirectoryseg,inputdirectoryscan,outputdirectoryscan

def check_dir(directory):
    if os.path.exists(directory)==False:
        print(directory,"doesn't exist")
        sys.exit(2)
    if os.path.isdir(directory)==False:
        print(directory, "is not a directory")
        sys.exit(2)
    if os.access(directory,os.W_OK)==False:
        print(directory, ": check your permissions")
        sys.exit(2)

def get_images(img_directory):
    temp=[]
    atleastone=False
    for iterator in os.listdir(img_directory):
            if(iterator.endswith(".mhd")):
                temp.append(img_directory+os.sep+os.path.basename(iterator))
                if(atleastone==False):
                    atleastone=True
            elif(iterator.endswith(".dcm")):
                temp.append(img_directory+os.sep+os.path.basename(iterator))
                if(atleastone==False):
                    atleastone=True
            elif(iterator.endswith(".nrrd")):
                temp.append(img_directory+os.sep+os.path.basename(iterator))
                if(atleastone==False):
                    atleastone=True
    if (atleastone==False):
        print("not the correct file format",img_directory)
    return sorted(temp)


if __name__ == "__main__":
    if(
    len(sys.argv)!=9 and len(sys.argv)!=11 and len(sys.argv)!=10 and len(sys.argv)!=13
    ):
        if(len(sys.argv)!=2):
            print("number of arguments = ",len(sys.argv),"oops, use -h for help ")
            sys.exit(2)
    log=False
    inputdirectoryseg,outputdirectoryseg,inputdirectoryscan,outputdirectoryscan=main(sys.argv[1:])
    image_directoryseg=get_images(inputdirectoryseg)
    image_directoryscan=get_images(inputdirectoryscan)
    try:
        for imageseg_ in image_directoryseg:
#for imagescan_ in image_directoryscan
            for imagescan_ in image_directoryscan:
                if islobe==True:
                    if airways_to_lobes!=True:
                        if(str(os.path.basename(imageseg_)).replace("_LobeSegmentation.nrrd","")!=str(os.path.basename(imagescan_)).replace(".mhd","")):
                            continue
                    else:
                        if(str(os.path.basename(imageseg_)).replace("_LobeSegmentation.nrrd","")!=str(os.path.basename(imagescan_)).replace("_airwaySegmentation.seg.nrrd","")):
                            continue
                else:
                    if(str(os.path.basename(imageseg_)).replace("_airwaySegmentation.seg.nrrd","")!=str(os.path.basename(imagescan_)).replace(".mhd","")):
                        continue
                imageseg=sitk.ReadImage(imageseg_)
                print(imageseg_,",")
                print(imageseg.GetSize(),",")
                if debug:
                    print("displaying original segmentation",str(os.path.basename(imageseg_)))
                    myshow(imageseg)
                if islobe==True:
                    imageseg = sitk.GetArrayFromImage(imageseg)
                    imageseg = np.where(imageseg>=4, 1, 0)
                    imageseg = sitk.GetImageFromArray(imageseg)
                    imageseg.CopyInformation(sitk.ReadImage(imageseg_))
                if debug:
                        get_hist(imageseg)
                label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
                label_shape_filter.Execute(imageseg)
                bounding_box = label_shape_filter.GetBoundingBox(1) #-1 due to binary nature of threshold
                imagescan=sitk.ReadImage(imagescan_)
                print(imagescan_,",")
                print(imagescan.GetSize(),",")
                roi_seg = sitk.RegionOfInterest(imageseg, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
                print(roi_seg.GetSize(),",")
               # try: 
                roi_scan=sitk.RegionOfInterest(imagescan, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
               # except RuntimeError as e:
               #     #if region of interest is outside, check and crop with original dims on the region outside of the range
               #     correct_roi=[]
               #     if roi_seg.GetSize()[0]>imagescan.GetSize()[0]:
               #         correct_roi.append(imagescan.GetSize()[0])
               #     else:
               #         correct_roi.append(roi_seg.GetSize()[0])
                    
               #     if roi_seg.GetSize()[1]>imagescan.GetSize()[1]:
               #         correct_roi.append(imagescan.GetSize()[1])
               #     else:
               #         correct_roi.append(roi_seg.GetSize()[1])
                    
               #     if roi_seg.GetSize()[2]>imagescan.GetSize()[2]:
               #         correct_roi.append(imagescan.GetSize()[2])
               #     else:
               #         correct_roi.append(roi_seg.GetSize()[2])
                    #repeat this for all dimensions
               #     bounding_box=tuple(correct_roi)
               #     roi_scan=sitk.RegionOfInterest(imagescan,bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
                    #if another error occurs it will be caught by the outside except, otherwise the error will be passed 
               #     print("runtime error was passed: ",e)
               #     pass
               # print(roi_scan.GetSize(),"\n")
                if airways_to_lobes==False:
                    sitk.WriteImage(roi_seg,str(outputdirectoryseg)+os.path.sep+os.path.basename(imageseg_))
                sitk.WriteImage(roi_scan,str(outputdirectoryscan)+os.path.sep+os.path.basename(imagescan_))
                break
                if debug:
                    print("displaying cropped segmentation",str(os.path.basename(imageseg_)))
                    print(str(bounding_box),": value written to crop_csv")
                    myshow(roi_seg)
                    print("displaying cropped scan",str(os.path.basename(imagescan_)))
                    myshow(roi_scan)
                    myshow(imagescan)
  #.copyinformation after converting to numpy array to avoid meta data loss     
    except RuntimeError as e:
        #do something useful
        print ("runtime error ",str(e))
        pass
else:
    print("command line parse error")


if (log):
    print("There were some errors, please check the log.txt file \n")
#df=pandas.DataFrame(bounding_box)
#df.to_csv('crop_values.csv',header=[os.path.basename(image_)[-8:-4]],index=False,columns=int(image_directory.index(image_)))
#for the lobes 
