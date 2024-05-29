from datasets import ChangeDetection
from palette import color_map
from pspnet import BaseNet
from params import *
from processor.TiffImageProcessor import TiffImageProcessor
from processor.ImageStitcher import ImageStitcher
from processor.ImageProcessor import ImageProcessor
from processor.change_shape import change_shape
from processor.down_image import MapDownloader
from processor.regularization import BuildingFootprintRegularization

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
import os

def main():
    testset = ChangeDetection(result, mode="test")
    testloader = DataLoader(testset, batch_size=2, shuffle=False,
                            pin_memory=True, num_workers=0, drop_last=False)

    model = BaseNet(6, lightweight)
    model.load_state_dict(torch.load('outdir/models/29_7.84.pth', map_location='cpu'), strict=True)

    model.eval()

    cmap = color_map()

    tbar = tqdm(testloader)

    with torch.no_grad():
        for k, (img1, img2, id) in enumerate(tbar):
            out1, out2, out_bin = model(img1, img2)

            out1 = torch.argmax(out1, dim=1)
            out2 = torch.argmax(out2, dim=1)
            out_bin = torch.argmax(out_bin, dim=1)
            out1[out_bin == 0] = 0
            out2[out_bin == 0] = 0
            out1 = out1.cpu().numpy()
            out2 = out2.cpu().numpy()
            out_bin = out_bin.cpu().numpy()

            for i in range(out1.shape[0]):
                mask = Image.fromarray(out1[i].astype(np.uint8), mode="P")
                mask.putpalette(cmap)
                mask.save(result+"/label1/" + id[i])

                mask = Image.fromarray(out2[i].astype(np.uint8), mode="P")
                mask.putpalette(cmap)
                mask.save(result+"/label2/" + id[i])

                mask = Image.fromarray(out_bin[i].astype(np.uint8), mode="P")
                mask.putpalette(cmap)
                mask.save(result+"/label3/" + id[i])

def download_and_process_map(url, min_lat, max_lat, min_lon, max_lon, zoom_level, image,data):
    map_downloader = MapDownloader(url, min_lat, max_lat, min_lon, max_lon, zoom_level)
    result_image = map_downloader.download_tiles()
    
    lon_min, lat_min, lon_max, lat_max = map_downloader.num2deg3857(*map_downloader.deg2num3857(min_lon, min_lat))
    lon_min1, lat_min1, lon_max1, lat_max1 = map_downloader.num2deg3857(*map_downloader.deg2num3857(max_lon, max_lat))
    
    map_downloader.create_georeferenced_tiff(result_image, lon_min, lat_min, lon_max1, lat_max1, image)
    processor = TiffImageProcessor(image, data, 512, 0)
    processor.tif_crop()
    processor = ImageProcessor()
    for file_name in os.listdir(data):
        input_file = os.path.join(data, file_name)
        processor.compress(input_file, input_file)

if __name__ == '__main__':
    if os.path.exists(result):
        shutil.rmtree(result)
    os.makedirs(result+"/im1")
    os.makedirs(result+"/im2")   
    os.makedirs(result+"/label1")
    os.makedirs(result+"/label2")
    os.makedirs(result+"/label3")
    download_and_process_map(url, min_lat, max_lat, min_lon, max_lon, zoom_level, image,result+"/im1")
    download_and_process_map(url_after, min_lat, max_lat, min_lon, max_lon, zoom_level, image_after,result+"/im2")
    main()
    # regularization = BuildingFootprintRegularization(result + "/jieyi/")
    # regularization.process_images()
    name,_ = os.path.splitext(image)
    stitcher = ImageStitcher(image,
                        result+"/label1",
                        name+"_predict.tif", 0)
    # Stitch images
    stitcher.tif_stitch()
    name_after,_ = os.path.splitext(image_after)
    stitcher = ImageStitcher(image_after,
                        result+"/label2",
                        name_after+"_predict.tif", 0)
    # Stitch images
    stitcher.tif_stitch()
    # change_shape(input_raster = name+"_predict.tif")
    shutil.rmtree(result)
