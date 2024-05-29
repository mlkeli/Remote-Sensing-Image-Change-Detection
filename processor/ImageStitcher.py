import os
try:
    import gdal
except:
    from osgeo import gdal
import numpy as np

class ImageStitcher:
    def __init__(self, ori_tif, tif_array_path, result_path, repetition_rate):
        self.ori_tif = ori_tif
        self.tif_array_path = tif_array_path
        self.result_path = result_path
        self.repetition_rate = repetition_rate

    # Function to save TIFF file
    def write_tiff(self, im_data, im_geotrans, im_proj, path):
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_Uint16
        else:
            datatype = gdal.GDT_Float32
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        elif len(im_data.shape) == 2:
            im_data = np.array([im_data])
            im_bands, im_height, im_width = im_data.shape
        # Create the file
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
        if dataset is not None:
            dataset.SetGeoTransform(im_geotrans)  # Write the affine transformation parameters
            dataset.SetProjection(im_proj)  # Write the projection
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset

    # Pixel coordinate and geographic coordinate affine transformation
    def coord_transf(self, Xpixel, Ypixel, GeoTransform):
        XGeo = GeoTransform[0] + GeoTransform[1] * Xpixel + Ypixel * GeoTransform[2]
        YGeo = GeoTransform[3] + GeoTransform[4] * Xpixel + Ypixel * GeoTransform[5]
        return XGeo, YGeo

    # Image stitching function
    def tif_stitch(self):
        RepetitionRate = float(self.repetition_rate)
        print("-------------------- Stitching Images -----------------------")
        dataset_img = gdal.Open(self.ori_tif)
        width = dataset_img.RasterXSize  # Get the number of rows and columns
        height = dataset_img.RasterYSize
        bands = dataset_img.RasterCount  # Get the number of bands
        proj = dataset_img.GetProjection()  # Get the projection information
        print(proj)
        geotrans = dataset_img.GetGeoTransform()  # Get the affine transformation matrix information
        print(geotrans)
        ori_img = dataset_img.ReadAsArray(0, 0, width, height)  # Get the data
        print("Number of bands:", bands)
        # Create an empty matrix
        if bands == 1:
            shape = [height, width]
        else:
            shape = [bands, height, width]
        result = np.zeros(shape, dtype='uint8')

        # Read the cropped images
        OriImgArray = []  # Create a queue
        NameArray = []
        imgList = os.listdir(self.tif_array_path)  # Read the folder
        imgList.sort(key=lambda x: int(x.split('.')[0]))  # Sort the files in the folder in numerical order
        for TifPath in imgList:
            dataset_img = gdal.Open(os.path.join(self.tif_array_path, TifPath))
            width_crop = dataset_img.RasterXSize  # Get the number of rows and columns
            height_crop = dataset_img.RasterYSize
            bands_crop = dataset_img.RasterCount  # Get the number of bands
            img = dataset_img.ReadAsArray(0, 0, width_crop, height_crop)  # Get the data
            OriImgArray.append(img)  # Add the image to the queue
            name = TifPath.split('.')[0]
            NameArray.append(name)
        print("Number of images read:", len(OriImgArray))

        # Calculate the number of image blocks in rows and columns
        RowNum = int((height - height_crop * RepetitionRate) / (height_crop * (1 - RepetitionRate)))
        ColumnNum = int((width - width_crop * RepetitionRate) / (width_crop * (1 - RepetitionRate)))
        sum_img = RowNum * ColumnNum + RowNum + ColumnNum + 1
        print("Number of image blocks in rows:", RowNum)
        print("Number of image blocks in columns:", ColumnNum)
        print("Total number of images:", sum_img)

        # Create an empty image
        if bands_crop == 1:
            shape_crop = [height_crop, width_crop]
        else:
            shape_crop = [bands_crop, height_crop, width_crop]
        img_crop = np.zeros(shape_crop)  # Create an empty matrix
        ImgArray = []
        count = 0
        for i in range(sum_img):
            img_name = i
            for j in range(len(OriImgArray)):
                if img_name == int(NameArray[j]):
                    image = OriImgArray[j]
                    count = count + 1
                    break
                else:
                    image = img_crop
            ImgArray.append(image)

        print("Number of images containing water bodies:", count)
        print("Total number of images in the list:", len(ImgArray))

        # Assign values
        num = 0
        for i in range(RowNum):
            for j in range(ColumnNum):
                if (bands == 1):
                    result[
                    int(i * height_crop * (1 - RepetitionRate)): int(i * height_crop * (1 - RepetitionRate)) + height_crop,
                    int(j * width_crop * (1 - RepetitionRate)): int(j * width_crop * (1 - RepetitionRate)) + width_crop] = \
                    ImgArray[num]
                else:
                    result[:,
                    int(i * height_crop * (1 - RepetitionRate)): int(i * height_crop * (1 - RepetitionRate)) + height_crop,
                    int(j * width_crop * (1 - RepetitionRate)): int(j * width_crop * (1 - RepetitionRate)) + width_crop] = \
                    ImgArray[num]
                num = num + 1
        # Last row
        for i in range(RowNum):
            if (bands == 1):
                result[
                int(i * height_crop * (1 - RepetitionRate)): int(i * height_crop * (1 - RepetitionRate)) + height_crop,
                (width - width_crop): width] = ImgArray[num]
            else:
                result[:,
                int(i * height_crop * (1 - RepetitionRate)): int(i * height_crop * (1 - RepetitionRate)) + height_crop,
                (width - width_crop): width] = ImgArray[num]
            num = num + 1
        # Last column
        for j in range(ColumnNum):
            num = num + 1
            if (bands == 1):
                result[(height - height_crop): height,
                int(j * width_crop * (1 - RepetitionRate)): int(j * width_crop * (1 - RepetitionRate)) + width_crop] = \
                ImgArray[num]
            else:
                result[:,
                (height - height_crop): height,
                int(j * width_crop * (1 - RepetitionRate)): int(j * width_crop * (1 - RepetitionRate)) + width_crop] = \
                ImgArray[num-1]

        if (bands == 1):
            result[(height - height_crop): height,
            (width - width_crop): width] = ImgArray[num]
        else:
            result[:,
            (height - height_crop): height,
            (width - width_crop): width] = ImgArray[num]

        self.write_tiff(result, geotrans, proj, self.result_path)

if __name__ == '__main__':
    stitcher = ImageStitcher(r"C:\Users\admin\Downloads\output_image66.tif",
                        r"C:\Users\admin\Downloads\tif",
                        r"C:\Users\admin\Downloads\output_image669.tif", 0)
    # Stitch images
    stitcher.tif_stitch()