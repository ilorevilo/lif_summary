# -*- coding: utf-8 -*-
"""
    lif_summary: simplified extraction of microscope images/ videos from a .lif-file
    written by Oliver Schneider (06/2021)
  
"""

#import sys
#from pptx import Presentation
#from pptx.util import Inches, Pt
#from pptx.dml.color import RGBColor
from itertools import zip_longest

import numpy as np
from readlif.reader import LifFile

from xml.etree import ElementTree as ET
#from xml import etree as et
from xml.dom import minidom

#from skimage.viewer import ImageViewer
from skimage import exposure
import skimage.io
#import PIL
from PIL import ImageDraw, ImageFont, Image

import cv2

import glob
import os
from pathlib import Path

import tifffile
import logging
import subprocess as sp
import shlex
import time
import tqdm
import sys
import colorama

colorama.init()
FFMPEG_BINARY = os.getenv("FFMPEG_BINARY", "ffmpeg-imageio")
#IMAGEMAGICK_BINARY = os.getenv("IMAGEMAGICK_BINARY", "auto-detect")
if FFMPEG_BINARY == "ffmpeg-imageio":
    from imageio.plugins.ffmpeg import get_exe

    FFMPEG_BINARY = get_exe()

class lif_summary:
    """
        class which contains all functions to extract images from lif file
    """

    def __init__(self, lif_file):
        """
            just specify path to lif-file in constructor
        """
        self.lif_path = Path(lif_file)
        
        self.filename = self.lif_path.stem       
        self.filename_full = self.lif_path.name
        self.outdir = self.lif_path.parent/self.filename      
        self.outdir.mkdir(parents=True, exist_ok=True)   
        
        self._release_logger()
        logging.basicConfig(filename=self.outdir/(self.filename+'_extractlog.log'), filemode='w', level=logging.DEBUG, format='%(message)s')
        
        print("#########################")
        print("reading file", self.filename_full)
        
        self.lifhandler = LifFile(self.lif_path)
        
        # categories under which images "series" will get sorted later
        self.export_entries = ["xy", "xyc", "xyz", "xyt"] # currently supported entrytypes for export
        self.nonexport_entries = ["xyct", "xycz", "xyzt", "xyczt", "envgraph", "MAF", "other"]
        self.grouped_img = {key:[] for key in self.export_entries}
        self.grouped_img.update( {key:[] for key in self.nonexport_entries})
        
        self._get_overview()
        self._log_overview()
        self.print_overview()
        self._write_xml()

    def _release_logger(self):
        """ 
            releases old logger instance, called upon init of new lif
            might be useful in e.g. jupyter where object not automatically released after export
        """
        
        logging.shutdown() #clear old logger
        logging.getLogger().handlers.clear()        

    def _build_query(self, imgentry, query=""):
        """
            constructs cmd to query specified element, takes also care if element is in subfolder
            imgentry = entry from img_list (=dict)
        """
        # procedure works, however a bit cumbersome... better to directly extract more param in readlif?
        
        path = imgentry["path"] # subfolders are nested here: projectname/subf1/subf2/
        sfolders = path.split("/")[1:-1] # split by / and take all except first and last
        name = imgentry["name"]
        
        elquery = 'Element/Children' # main entrypoint, all images are always children of main element
        for sfolder in sfolders:
            elquery = elquery + f'/Element[@Name="{sfolder}"]/Children' # build query of all subfolders
            
        elquery = elquery + f'/Element[@Name="{name}"]' # attach query for element with specified name
        query = elquery + query
        return query

    def _query_hist(self, imgentry):
        """
            reads out BlackValue and WhiteValue for specific imgentry
            take care, multichannel not implemented yet, will return values of first found entry
        """
        query = self._build_query(imgentry, "/Data/Image/Attachment/ChannelScalingInfo")
        
        rootel = self.lifhandler.xml_root
        blackval = float(rootel.find(query).attrib["BlackValue"])
        whiteval = float(rootel.find(query).attrib["WhiteValue"])
        
        return [blackval, whiteval]
    
    def _query_chan(self, imgentry):
        """
            reads out used contrast method + filter cube for specific imgentry
            returns as list where each item corresponds to channel
        """

        cquery = self._build_query(imgentry,"Data/Image/Attachment/"
            "ATLCameraSettingDefinition/WideFieldChannelConfigurator/WideFieldChannelInfo")
        rootel = self.lifhandler.xml_root 
        chan_els = rootel.findall(cquery)
        chaninfo = [chan_el.attrib["ContrastingMethodName"] + "_" + chan_el.attrib["FluoCubeName"] for chan_el in chan_els]            
        
        return chaninfo
        
    def _query_timestamp(self, imgentry):
        """ returns acquision date (= first timestamp) of selected entry """
        tsquery = self._build_query(imgentry,"Data/Image/TimeStampList")
        rootel = self.lifhandler.xml_root
        # try:
        ts = rootel.find(tsquery).text
        ts = ts.split(" ")[0]
        # except (AttributeError, TypeError):
        # return None
        
        # conversion adapted from bioformats
        stampLowStart = max(0, len(ts) - 8)
        stampHighEnd = max(0, stampLowStart)

        stampHigh = ts[0: stampHighEnd]
        stampLow = ts[stampLowStart:]
        
        low = int(stampLow,16)
        high = int(stampHigh,16)    

        ticks = (high << 32) | low
        ticks = ticks/10000
        
        COBOL_EPOCH = 11644473600000
        ts_unix = int(ticks - COBOL_EPOCH) # in ms

        return ts_unix
    
    def _log_img(self,imgentry):
        """
            logs currently exported imagentry to logfile
        """
        logging.warning(f"########## exporting entry {imgentry['idx']} ##########")
        for entry in ["name", "path", "bit_depth", "dims", "scale", 
                        "channels", "chaninfo", "Blackval", "Whiteval", "AcqTS"]:
            logging.warning(f'{entry}: %s', imgentry[entry])
    
    def _log_overview(self):
        """ logs info of entries added by _get_overview to logfile """
        
        logging.warning(f"########## entries found in file ##########")
        logging.warning(f"- entries which will be exported:")

        for imgtype in self.export_entries:
            imglist = self.grouped_img[imgtype]

            logging.warning(f'{imgtype}: {len(imglist)}')
            
        logging.warning(f"- entries whose export is not supported yet:")
        
        N_nonexported = 0
        for imgtype in self.nonexport_entries:
            imglist = self.grouped_img[imgtype]

            if len(imglist) > 0:
                N_nonexported += len(imglist)
                
            logging.warning(f'{imgtype}: {len(imglist)}')         

        if N_nonexported > 0:
            logging.warning(f"### {N_nonexported} entries won't be exported ###")

    
    def print_overview(self):
        """ prints overview of found entries, shows which ones will be exported """
    
        print("following entries found in file: ")
        print("- entries which will be exported:")
        
        for imgtype in self.export_entries:
            imglist = self.grouped_img[imgtype]

            col = '\033[0m'
            if len(imglist) > 0:
                col = '\033[92m'
                
            print(f'{col} {imgtype}: {len(imglist)}' + '\033[0m')   
        
        print("- entries whose export is not supported yet:")

        N_nonexported = 0
        for imgtype in self.nonexport_entries:
            imglist = self.grouped_img[imgtype]
            col = '\033[0m'
            if len(imglist) > 0:
                N_nonexported += len(imglist)
                col = '\033[93m'
                
            print(f'{col} {imgtype}: {len(imglist)}' + '\033[0m')   
        
        if N_nonexported > 0:
            print(f"\033[91m {N_nonexported} entries won't be exported \033[0m")
        
    def _get_overview(self):
        """
            extracts information of stored images from metadata
            fills dict self.grouped_img with dict for each imgentry
        """
        
        for idx, img in enumerate(self.lifhandler.image_list):
            
            # print(img)
            img["idx"] = idx # add index which is used to request frame
            img["chaninfo"] = self._query_chan(img)
            img["AcqTS"] = self._query_timestamp(img)
            img["Blackval"] = None
            img["Whiteval"] = None
            
            # check for special cases first
            if ('EnvironmentalGraph') in img["name"]:
                self.grouped_img["envgraph"].append(img)
                continue
            
            if ('Mark_and_Find') in img["path"]:
                self.grouped_img["MAF"].append(img)
                continue
                
            # check various dimensions to sort entries accordingly
            dimtuple = img["dims"]
            Nx, Ny, Nz, NT = dimtuple[0], dimtuple[1], dimtuple[2], dimtuple[3]
             # dimension tuple must be indexed with int, 0:x, 1:y, 2:z, 3:t, 4:m mosaic tile
            NC = img["channels"]

            # xy (simple image)
            if (Nz == 1 and NT == 1 and NC == 1):
                # print("entry is xy")
                self.grouped_img["xy"].append(img)

            # xyc (multichannel image)
            elif (Nz == 1 and NT == 1 and NC > 1):
                # print("entry is xyc")
                self.grouped_img["xyc"].append(img)

            # xyz (singlechannel zstack)
            elif (Nz > 1 and NT == 1 and NC == 1):
                # print("entry is xyz")
                self.grouped_img["xyz"].append(img)
                
            # xyt singlechannel video/ timelapse
            elif (Nz == 1 and NT > 1 and NC==1):
                # print("entry is xyt")
            
                img["fps"] = img["scale"][3]
                self.grouped_img["xyt"].append(img)
                
            # xyct (multichannel video/ timelapse)
            elif (Nz == 1 and NT > 1 and NC > 1):

                img["fps"] = img["scale"][3]
                self.grouped_img["xyct"].append(img)
                
            # xycz
            elif (Nz > 1 and NT == 1 and NC > 1):
                self.grouped_img["xycz"].append(img)
                
            # xyzt
            elif (Nz > 1 and NT > 1 and NC == 1):
                img["fps"] = img["scale"][3]
                self.grouped_img["xyzt"].append(img)

            # xyczt
            elif (Nz > 1 and NT > 1 and NC > 1):
                img["fps"] = img["scale"][3]
                self.grouped_img["xyczt"].append(img)                

            # add to category other if no previously checked category applies
            else:
                self.grouped_img["other"].append(img)
        
        # find blackval/whiteval (or even other param if desired) for xy and xyt-images
        for cat in ["xy","xyt"]:
            imglist = self.grouped_img[cat]
            
            for img in imglist:
                black_val, white_val = self._query_hist(img)
                img["Blackval"] = black_val
                img["Whiteval"] = white_val
    
    def _write_xml(self):
        """
            writes metadata of lif-file in pretty xml
        """

        xmlstr = minidom.parseString(ET.tostring(self.lifhandler.xml_root)).toprettyxml(indent="   ")
        
        fname = self.outdir/(self.filename + "_meta.xml")
        with open(fname, "w") as f:
            f.write(xmlstr)        
        #https://stackoverflow.com/questions/56682486/xml-etree-elementtree-element-object-has-no-attribute-write
        
    def export_xy(self, min_rangespan=0.2, max_clipped=0.2):
        """
            exports all xy image entries:
            - raw export: tif 
            - compressed export (jpg, scaled to blackval/whiteval which was set during acquisition 
                with burned in title + scale bar)
        """
        
        # check if entries to export        
        if len(self.grouped_img["xy"]) == 0:
            return
        
        #### raw export folder
        rawfolder = self.outdir/"Images_xy"/"raw"
        rawfolder.mkdir(parents=True, exist_ok=True)
        #### compressed jpg export folder
        compfolder = self.outdir/"Images_xy"/"compressed"
        compfolder.mkdir(parents=True, exist_ok=True)          
        
        # iterate all images
        for imgentry in self.grouped_img["xy"]:
            
            self._log_img(imgentry)
            img_idx = imgentry["idx"]
            img_name = imgentry["name"]
            print(f"exporting image {img_name}")# with meta: \n {imgentry}")
            """
            # option to concatenate subfolders into filename
            path = imgentry["path"] # subfolders are nested here: projectname/subf1/subf2/
            sfolders = path.split("/")[1:-1] # split by / and take all except first and last
            sfolders.append(img_name)
            img_name = ("_".join(sfolders))
            """        
            
            imghandler = self.lifhandler.get_image(img_idx)
            img = imghandler.get_frame(z=0, t=0, c=0)
            img_np = np.array(img)

            resolution_mpp = 1.0/imgentry["scale_n"][1] # unit should be pix per micron of scale_n
            
            imgpath = rawfolder/(img_name+".tif")
            self.save_single_tif(img_np, imgpath, resolution_mpp)
            
            # compressed export:
            # scale images to 8bit, add scalebar + title, save as jpg in orig resolution
            bit_resolution = imgentry["bit_depth"][0]
            img_scale = 2**bit_resolution - 1
            
            # image_adj_contrast = self.adj_contrast(img_np, imgentry["Blackval"]*img_scale, imgentry["Whiteval"]*img_scale)
            vmin, vmax = self.check_contrast(img_np, 
                imgentry["Blackval"]*img_scale, imgentry["Whiteval"]*img_scale,
                min_rangespan=min_rangespan, max_clipped=max_clipped)
            image_adj_contrast = exposure.rescale_intensity(img_np, in_range=(vmin, vmax)) # stretch min/max
            img_8 = cv2.convertScaleAbs(image_adj_contrast,alpha=(255.0/img_scale))
            
            labeled_image = self.plot_scalebar(img_8, resolution_mpp, img_name)
            
            imgpath_jpg = compfolder/(img_name+".jpg")
            skimage.io.imsave(imgpath_jpg, labeled_image)
        
    def export_xyz(self):
        """
            exports all xyz image entries (=zstacks)
            - raw export: tif 
            - compressed export: none
        """ 

        # check if entries to export
        if len(self.grouped_img["xyz"]) == 0:
            return

        #### raw export folder
        rawfolder = self.outdir/"Images_xyz"
        rawfolder.mkdir(parents=True, exist_ok=True)
        
        # iterate all images
        for imgentry in self.grouped_img["xyz"]:

            self._log_img(imgentry)
            resolution_mpp = 1.0/imgentry["scale_n"][1] # unit should be pix per micron of scale_n
            img_idx = imgentry["idx"]
            img_name = imgentry["name"]
            print(f"exporting zstack {img_name}")# with meta: \n {imgentry}")
            imghandler = self.lifhandler.get_image(img_idx)   
            
            # get correct z-spacing dz
            # take care! might need to be adjusted if reader.py changes
            z_spacing = imgentry["scale_n"][3] # planes per micron
            Nz = imgentry["dims"][2]
            total_z = Nz/z_spacing
            dz = total_z/(Nz-1)
            
            dzstring = f"-dz_{dz:.2f}_um".replace(".","_") # write plane spacing into foldername such that it's easily accessible
            stackfolder = rawfolder/(img_name+dzstring) # create overall folder for zstack
            stackfolder.mkdir(parents=True, exist_ok=True)
            
            Nplanes = imgentry["dims"][2]
            for plane in tqdm.tqdm(np.arange(Nplanes), desc="Plane",
                    file = sys.stdout, position = 0, leave=True):            
                
                img = imghandler.get_frame(z=plane, t=0, c=0)
                img_np = np.array(img)                
                
                planepath = stackfolder/f"{img_name}-{plane:04d}.tif"  # series_name+"-{:04d}.tif".format(plane))
                self.save_single_tif(img_np,planepath, resolution_mpp)      
        
    def export_xyc(self):
        """
            exports all xyc image entries (=multichannel images)
            - raw export: tif 
            - compressed export: none            
        """
        
        # check if entries to export        
        if len(self.grouped_img["xyc"]) == 0:
            return
            
        #### raw export folder
        rawfolder = self.outdir/"Images_xyc"
        rawfolder.mkdir(parents=True, exist_ok=True)        

        # iterate all images
        for imgentry in self.grouped_img["xyc"]:
            
            self._log_img(imgentry)
            resolution_mpp = 1.0/imgentry["scale_n"][1] # unit should be pix per micron of scale_n
            img_idx = imgentry["idx"]
            img_name = imgentry["name"]
            imgpath = rawfolder/(img_name+".tif")
            print(f"exporting multichannel {img_name}")# with meta: \n {imgentry}")

            
            imghandler = self.lifhandler.get_image(img_idx)
            channel_list = [np.array(img) for img in imghandler.get_iter_c(t=0, z=0)]

            img_xyc = np.array(channel_list)
            self.save_single_tif(img_xyc, imgpath, resolution_mpp, photometric = 'minisblack')            

    def export_xyt(self, min_rangespan=0.2, max_clipped=0.2):
        """
            exports all xyt image entries (=video/ timelapse entries)
            directly pipes frames to ffmpeg
            - large export: .mp4 in full resolution, low compression
            - small export: .mp4, longest side scaled to 1024, include scalebar          
        """
        # check if entries to export        
        if len(self.grouped_img["xyt"]) == 0:
            return        
        
        #### hq export folder
        lgfolder = self.outdir/"Videos"/"lg"
        lgfolder.mkdir(parents=True, exist_ok=True)
        #### compressed jpg export folder
        smfolder = self.outdir/"Videos"/"sm"
        smfolder.mkdir(parents=True, exist_ok=True)

        # iterate all entries
        for imgentry in self.grouped_img["xyt"]:
       
            self._log_img(imgentry)
            resolution_mpp = 1.0/imgentry["scale_n"][1]
            img_idx = imgentry["idx"]
            img_name = imgentry["name"]
            fps = imgentry['fps']
            Nx, Ny, NT = imgentry["dims"][0], imgentry["dims"][1], imgentry["dims"][3]
            Nmax_sm = 1024 # longest side of small video
            codec_lg, codec_sm = 'libx264', 'libx264'
            crf_lg, crf_sm = 17, 23

            #rescale smaller video such that longest side = 1024
            # prevent upscaling
            Nlong = max(Nx, Ny)
            scalingfactor = float(Nmax_sm)/Nlong
            if scalingfactor > 1.0: # don't allow upscaling
                scalingfactor = 1.0
            #print("scaling", scalingfactor)
            Nx_sm, Ny_sm = int(Nx*scalingfactor), int(Ny*scalingfactor)

            scalebar = self.create_scalebar(Nx_sm,resolution_mpp/scalingfactor) #create scalebar for small vid
            scale_width_px = scalebar.shape[0]
            scale_height_px = scalebar.shape[1]
            
            print(f"exporting video {img_name}")# with meta: \n {imgentry}")
            #print("resolution of small video:", Nx_sm, Ny_sm)            
            imghandler = self.lifhandler.get_image(img_idx)

            # export of both vids simultaneously, get infos to start ffmpeg subprocess
            
            resinfo_lg = f'resolution_xy={resolution_mpp:.6f}_mpp'.replace(".","_")
            resinfo_sm = f'resolution_xy={resolution_mpp/scalingfactor:.6f}_mpp'.replace(".","_") # correct by scalingfactor
            # stores resolution info in metadata (in category comment) for quick access from videofile
            path_lg = str(lgfolder/(img_name+"_lg.mp4")) # string needed for input to ffmpeg cmd
            path_sm = str(smfolder/(img_name+"_sm.mp4")) # string needed for input to ffmpeg cmd        
            sizestring_lg = f'{Nx}x{Ny}' # e.g. 1024x1024, xsize x ysize, todo: check if order correct
            sizestring_sm = f'{Nx_sm}x{Ny_sm}'

            startt = time.time() #for quick check of exporttimes
            
            # write video via pipe to ffmpeg-stream, start process here
            # solution from https://stackoverflow.com/questions/61260182/how-to-output-x265-compressed-video-with-cv2-videowriter
            process_lg = sp.Popen(shlex.split(f'"{FFMPEG_BINARY}" -y -s {sizestring_lg} '
                f'-pixel_format gray8 -f rawvideo -r {fps} -i pipe: -vcodec {codec_lg} '
                f'-pix_fmt yuv420p -crf {crf_lg} -metadata comment="{resinfo_lg}" "{path_lg}"'), stdin=sp.PIPE,
                stderr=sp.DEVNULL) # supress ffmpeg output to console
             
            # directly create process for export of smaller vid -> img has to be pulled only once
            process_sm = sp.Popen(shlex.split(f'"{FFMPEG_BINARY}" -y -s {sizestring_sm} '
                f'-pixel_format gray8 -f rawvideo -r {fps} -i pipe: -vcodec {codec_sm} '
                f'-pix_fmt yuv420p -crf {crf_sm} -metadata comment="{resinfo_sm}" "{path_sm}"'), stdin=sp.PIPE,
                stderr=sp.DEVNULL) #
            
            # check correct exposure scaling on one frame (image at half videolength)
            # save frame also as tif in full res for later access
            Nmean = int(imgentry["dims"][3]/2) - 1# idx of mean frame ~ at half of Nframes
            mimg = imghandler.get_frame(z=0, t=Nmean, c=0)
            img_np = np.array(mimg)
            
            bit_resolution = imgentry["bit_depth"][0]
            img_scale = 2**bit_resolution - 1               
            vmin, vmax = self.check_contrast(img_np, 
                imgentry["Blackval"]*img_scale, imgentry["Whiteval"]*img_scale,
                min_rangespan=min_rangespan, max_clipped=max_clipped)
            vmin8, vmax8 = vmin*(255.0/img_scale), vmax*(255.0/img_scale) # adjust to 8bit
            # print(imgentry["Blackval"], imgentry["Whiteval"])
            # print(vmin8, vmax8)
            # logging.warning(f'set vmin8, vmax8 to {vmin8}, {vmax8}')
            
            # save single tiff in full size
            stillpath = self.outdir/"Videos"/"tifstills"
            stillpath.mkdir(parents=True, exist_ok=True)
            self.save_single_tif(img_np, stillpath/(img_name+".tif"), resolution_mpp)            
            
            # kwarg info:
            # -y: overwrite wo. asking
            # -s: size
            # -pixel_format: bgr24 was set... use gray8 for 8bit grayscale
            # -f: "Force input or output file format. -> here set to raw stream"
            # -r: framerate, can be used for input and output stream, here only input specified -> output will be same
            # -i: pipe
            for frame in tqdm.tqdm(imghandler.get_iter_t(c=0, z=0), desc="Frame",
                    file = sys.stdout, position = 0, leave=True, total=NT):
                img_np = np.array(frame)
                img8 = cv2.convertScaleAbs(img_np,alpha=(255.0/img_scale)) # scale to 8bit range
                img_scaled = exposure.rescale_intensity(img8, in_range=(vmin8, vmax8)) # stretch min/max
                
                img_sm = cv2.resize(img_scaled, (Nx_sm,Ny_sm)) # scale down small vid
                img_sm[-1-scale_width_px:-1,-1-scale_height_px:-1] = scalebar # add scalebar
                
                process_lg.stdin.write(img_scaled.tobytes())
                process_sm.stdin.write(img_sm.tobytes())
                
            for process in [process_lg, process_sm]:
                process.stdin.close()
                process.wait()
                process.terminate()
            
            print("video export finished in", time.time()-startt, "s")

    def export_all(self):
        """ 
            exports xy, xyc, xyz, xyt entries (all currently supported export options)
            by calling individual exportfunctions
        """

        self.export_xy(min_rangespan = 0.2, max_clipped = 0.6)
        self.export_xyz()
        self.export_xyc()
        self.export_xyt(min_rangespan = 0.2, max_clipped = 0.6)

    def save_single_tif(self, image, path, resolution_mpp, photometric = None, compress = None):
        """
            saves single imagej-tif into specified folder
            uses resolution_mpp to indicate resolution in x and y dimensions in microns per pixel
        """
                
        resolution_ppm = 1/resolution_mpp # convert micron per pixel to pixel per micron
        metadata = {'unit': 'um'}
        
        if photometric == None:
            tifffile.imsave(path, image, imagej=True, resolution=(resolution_ppm, resolution_ppm), 
            metadata=metadata, compress = compress) #what if ppm is different in x and y?
        else:
            tifffile.imsave(path, image, imagej=True, resolution=(resolution_ppm, resolution_ppm), 
            metadata=metadata, photometric = photometric, compress = compress)
    
    def check_contrast(self, image, vmin=None, vmax =None, min_rangespan=0.2, max_clipped=0.2):
        """
            checks if desired scaling range between vmin and vmax yields to a reasonable
            intensity range (< max_clipped (20 % default) of image over/underexposed, 
            image spans > min_rangespan (20 % default) of range)
            adjusts vmin and vmax to 0.2 - 99.8 percentile if not
        """
        
        # check first if contrast is somehow alright
        
        # check spanwidth of image vs. spanwidth of defined interval
        # if rangespan small -> low contrast
        # rangespan can also be alright but values shifted -> over/ underxposure
        # -> check both
        imglimits = np.percentile(image, [5, 95])
        rangespan = (imglimits[1] - imglimits[0]) / (vmax - vmin) 
        
        # check fraction of px outside defined interval (max. 1 = all)
        # outside px are over/underexposed
        px_clipped = ((image < vmin) | (image > vmax)).sum()/ image.size
        if ((px_clipped > max_clipped) or (rangespan < min_rangespan)):
            print('\033[96m' + "extracted histogram scaling (blackval/ whiteval) would " 
                "correspond to an over/underexposure of > 20 % of the image "
                "or the image would span < 20 % of chosen range"
                "-> switching to automatic rescaling to range from 0.2 - 99.8 percentile" +
                '\033[0m')
            logging.warning(f'adjusting contrast range to {vmin}, {vmax}')
            #print(f"vmin {vmin}, vmax {vmax}, immin {imglimits[0]}, immax {imglimits[1]}")
            vmin, vmax = None, None
        
        if None in (vmin, vmax):
            vmin = np.percentile(image, 0.2)
            vmax = np.percentile(image, 99.8)      

        return vmin, vmax
        
    def create_scalebar(self, dimX_px, microns_per_pixel):
        """
            creates scalebar as np array which then can be transferred to image
        """
        
        scale_values = [1,2,5,10,15,20,30,40,50,70,100,150,200,300,400,500,700,1000]
        initial_scale_length = dimX_px * 0.2 * microns_per_pixel
        
        text_height_px = int(round(dimX_px * 0.05))
        drawfont = ImageFont.truetype("arial.ttf", text_height_px)
        
        scale_length_microns = min(scale_values, key=lambda x:abs(x-initial_scale_length))    # pick nearest value
        scale_caption = str(scale_length_microns) + " µm"
        scale_length_px = scale_length_microns / microns_per_pixel
        scale_height_px = dimX_px * 0.01
        
        bg_square_spacer_px = scale_length_px * 0.07
        
        bg_square_length_px = int(round(scale_length_px + 2 * bg_square_spacer_px))
        bg_square_height_px = int(round(text_height_px + scale_height_px + 2 * bg_square_spacer_px))
        
        scalebar = Image.new("L", (bg_square_length_px, bg_square_height_px), "white")
        draw = ImageDraw.Draw(scalebar)
        
        w_caption, h_caption = draw.textsize(scale_caption, font = drawfont)        
        
        draw.rectangle(((0, 0), (bg_square_length_px, bg_square_height_px)), fill="black")
        draw.rectangle(((bg_square_spacer_px, bg_square_height_px - bg_square_spacer_px - scale_height_px), (bg_square_length_px - bg_square_spacer_px, bg_square_height_px - bg_square_spacer_px)), fill="white")
        draw.text((bg_square_spacer_px + bg_square_length_px/2 - w_caption/2, bg_square_spacer_px/2), scale_caption, font = drawfont, fill="white")
        
        output_scalebar = np.array(scalebar)
        return output_scalebar

    #todo: reorganize such that cmds are not repeated in create_scalebar...
    def plot_scalebar(self, input_image, microns_per_pixel, image_name = None):
        """
            plots scalebar + title onto image if desired
            scalebar is only plotted if image width > 800 px
            image input: np array
        """
        
        image_scalebar = Image.fromarray(input_image) # Image is PIL.Image
        #np.uint8(input_image*255)
        
        dimX_px = input_image.shape[1]
        dimY_px = input_image.shape[0]
        initial_scale_length = dimX_px * 0.2 * microns_per_pixel
        
        text_height_px = int(round(dimY_px * 0.05))
        
        scale_values = [1,2,5,10,15,20,30,40,50,70,100,150,200,300,400,500,700,1000]
        drawfont = ImageFont.truetype("arial.ttf", text_height_px)
        
        scale_length_microns = min(scale_values, key=lambda x:abs(x-initial_scale_length))    # pick nearest value
        scale_caption = str(scale_length_microns) + " µm"
        
        draw = ImageDraw.Draw(image_scalebar)
        w_caption, h_caption = draw.textsize(scale_caption, font = drawfont)
        
        scale_length_px = scale_length_microns / microns_per_pixel
        scale_height_px = dimY_px * 0.01
        
        bg_square_spacer_px = scale_length_px * 0.07
        
        bg_square_length_px = scale_length_px + 2 * bg_square_spacer_px
        bg_square_height_px = text_height_px + scale_height_px + 2 * bg_square_spacer_px

        if dimX_px > 800:
            #print(dimX_px - bg_square_length_px, dimX_px - bg_square_height_px)
            draw.rectangle(((dimX_px - bg_square_length_px, dimY_px - bg_square_height_px), (dimX_px, dimY_px)), fill="black")
            draw.rectangle(((dimX_px - bg_square_length_px + bg_square_spacer_px, dimY_px - bg_square_spacer_px - scale_height_px), (dimX_px - bg_square_spacer_px, dimY_px - bg_square_spacer_px)), fill="white")
            draw.text((dimX_px - bg_square_length_px + bg_square_spacer_px + bg_square_length_px/2 - w_caption/2, dimY_px - bg_square_height_px + bg_square_spacer_px/2), scale_caption, font = drawfont, fill="white")# scale_caption.decode('utf8')

            # burn title if provided
            if image_name != None:
                title_height_px = int(round(dimY_px * 0.05))
                drawfont = ImageFont.truetype("arial.ttf", title_height_px)
                draw.rectangle(((0,0),(dimX_px,title_height_px*1.2)), fill = "black")
                draw.text((0,0),image_name, font = drawfont, fill = "white")
        
        output_image = np.array(image_scalebar)
                
        return output_image

    def create_ppt_summary(self):
        """
            creates ppt with all exported images
        """
        
        print("creating ppt-summary")
        
        # take always n elements from list as list, helpfunction
        def grouper(iterable, n, fillvalue=None):
            args = [iter(iterable)] * n
            return zip_longest(*args, fillvalue=fillvalue)
        
        # prepare presentation
        prs = Presentation()
        title_slide_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(title_slide_layout)
        title = slide.shapes.title
        subtitle = slide.placeholders[1]

        title.text = str(self.filename)
        subtitle.text = "lif-summary"

        ###############################
        # constants for 2 x 3 layout

        Pt_per_cm = 72.0/2.54

        slide_width = Pt (25.4 * Pt_per_cm)
        slide_height = Pt (19.05 * Pt_per_cm)

        headspace = Pt(30)
        image_height = Pt(200)

        d_horizontal = (slide_width-3*image_height)/4
        d_vertical = (slide_height-headspace-2*image_height)/3

        ##############################
        
        # pick 6 images
        
        for images_6group in grouper(self.categorized_series['img_simple'], 6, None):

            #add new slide
            blank_slide_layout = prs.slide_layouts[6]
            slide = prs.slides.add_slide(blank_slide_layout)
            
            #iterate through rows, columns
            for rowindex in range(2):
                for columnindex in range(3):
                    
                    image_index = rowindex * 3 + columnindex
                    if(images_6group[image_index]) != None:
                        image_name = images_6group[image_index]['name']
                        imagepath = os.path.join(self.filename,"compressed","images",image_name + ".jpg")
                        pic = slide.shapes.add_picture(imagepath, d_horizontal * (columnindex + 1) + image_height * columnindex, headspace + d_vertical * (rowindex + 1) + image_height * rowindex, height=image_height)
        
        """
        # pick videos, experimental, works but stillimage is loudspeaker -> create individual stillimage
        for video in self.categorized_series['img_multiT']:
            blank_slide_layout = prs.slide_layouts[6]
            slide = prs.slides.add_slide(blank_slide_layout)
            videopath = os.path.join(self.filename,"compressed","videos",video['name'] + ".mp4")
            slide.shapes.add_movie(videopath,d_vertical,d_vertical,slide_height-2*d_vertical,slide_height-2*d_vertical)
        """
        
        outputpath = os.path.join(str(self.filename),str(self.filename) + '_summary.pptx')
        prs.save(outputpath)
        
def main():

    inputlifs = glob.glob("*.lif") # get all .lif-files in current folder
    #inputlifs = [inputfile]    # if you just want to read one file
    
    # export each file
    for inputfile in inputlifs:
        try:
            new_summary = lif_summary(inputfile)
            # add option again to check if folder already exists?
            new_summary.export_all()
            #new_summary.create_ppt_summary()
        except Exception as ex:
            logging.exception('Error in export of lif-file')
            print( '\033[91m' + 'Error occured in export of current lif-file, '
            'check log for detailed info. Skipping to next file' + '\033[0m')   
    
if __name__ == "__main__":
    main()