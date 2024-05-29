import os
try:
    import gdal
except:
    from osgeo import gdal
import numpy as np


class TiffImageProcessor:
    def __init__(self, tif_path, save_path, crop_size, repetition_rate):
        self.tif_path = tif_path
        self.save_path = save_path
        self.crop_size = int(crop_size)
        self.repetition_rate = float(repetition_rate)

    def write_tiff(self, im_data, im_geotrans, im_proj, path):
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        elif len(im_data.shape) == 2:
            im_data = np.array([im_data])
            im_bands, im_height, im_width = im_data.shape

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
        if dataset is not None:
            dataset.SetGeoTransform(im_geotrans)
            dataset.SetProjection(im_proj)
        
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset

    def coord_transf(self, x_pixel, y_pixel, geo_transform):
        x_geo = geo_transform[0] + geo_transform[1] * x_pixel + y_pixel * geo_transform[2]
        y_geo = geo_transform[3] + geo_transform[4] * x_pixel + y_pixel * geo_transform[5]
        return x_geo, y_geo

    def tif_crop(self):
        print("--------------------裁剪影像-----------------------")
        dataset_img = gdal.Open(self.tif_path)
        if dataset_img is None:
            print(self.tif_path + "文件无法打开")

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        width = dataset_img.RasterXSize
        height = dataset_img.RasterYSize
        bands = dataset_img.RasterCount
        print("行数为：", height)
        print("列数为：", width)
        print("波段数为：", bands)

        proj = dataset_img.GetProjection()
        geotrans = dataset_img.GetGeoTransform()
        img = dataset_img.ReadAsArray(0, 0, width, height)

        row_num = int((height - self.crop_size * self.repetition_rate) / (self.crop_size * (1 - self.repetition_rate)))
        column_num = int((width - self.crop_size * self.repetition_rate) / (self.crop_size * (1 - self.repetition_rate)))
        print("裁剪后行影像数为：", row_num)
        print("裁剪后列影像数为：", column_num)

        new_name = len(os.listdir(self.save_path))

        for i in range(row_num):
            for j in range(column_num):
                if bands == 1:
                    cropped = img[
                              int(i * self.crop_size * (1 - self.repetition_rate)): int(i * self.crop_size * (1 - self.repetition_rate)) + self.crop_size,
                              int(j * self.crop_size * (1 - self.repetition_rate)): int(j * self.crop_size * (1 - self.repetition_rate)) + self.crop_size]
                else:
                    cropped = img[:,
                              int(i * self.crop_size * (1 - self.repetition_rate)): int(i * self.crop_size * (1 - self.repetition_rate)) + self.crop_size,
                              int(j * self.crop_size * (1 - self.repetition_rate)): int(j * self.crop_size * (1 - self.repetition_rate)) + self.crop_size]

                XGeo, YGeo = self.coord_transf(int(j * self.crop_size * (1 - self.repetition_rate)),
                                              int(i * self.crop_size * (1 - self.repetition_rate)),
                                              geotrans)
                crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])

                self.write_tiff(cropped, crop_geotrans, proj, self.save_path + "/%d.png" % new_name)
                ds = self.save_path + "/%d.png" % new_name
                new_name = new_name + 1

        for i in range(row_num):
            if bands == 1:
                cropped = img[int(i * self.crop_size * (1 - self.repetition_rate)): int(i * self.crop_size * (1 - self.repetition_rate)) + self.crop_size,
                          (width - self.crop_size): width]
            else:
                cropped = img[:,
                          int(i * self.crop_size * (1 - self.repetition_rate)): int(i * self.crop_size * (1 - self.repetition_rate)) + self.crop_size,
                          (width - self.crop_size): width]

            XGeo, YGeo = self.coord_transf(width - self.crop_size,
                                          int(i * self.crop_size * (1 - self.repetition_rate)),
                                          geotrans)
            crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])

            self.write_tiff(cropped, crop_geotrans, proj, self.save_path + "/%d.png" % new_name)
            ds = self.save_path + "/%d.png" % new_name
            new_name = new_name + 1

        for j in range(column_num):
            if bands == 1:
                cropped = img[(height - self.crop_size): height,
                          int(j * self.crop_size * (1 - self.repetition_rate)): int(j * self.crop_size * (1 - self.repetition_rate)) + self.crop_size]
            else:
                cropped = img[:,
                          (height - self.crop_size): height,
                          int(j * self.crop_size * (1 - self.repetition_rate)): int(j * self.crop_size * (1 - self.repetition_rate)) + self.crop_size]

            XGeo, YGeo = self.coord_transf(int(j * self.crop_size * (1 - self.repetition_rate)),
                                          height - self.crop_size,
                                          geotrans)
            crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])

            self.write_tiff(cropped, crop_geotrans, proj, self.save_path + "/%d.png" % new_name)
            ds = self.save_path + "/%d.png" % new_name
            new_name = new_name + 1

        if bands == 1:
            cropped = img[(height - self.crop_size): height,
                      (width - self.crop_size): width]
        else:
            cropped = img[:,
                      (height - self.crop_size): height,
                      (width - self.crop_size): width]

        XGeo, YGeo = self.coord_transf(width - self.crop_size,
                                     height - self.crop_size,
                                     geotrans)
        crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])

        self.write_tiff(cropped, crop_geotrans, proj, self.save_path + "/%d.png" % new_name)
        ds = self.save_path + "/%d.png" % new_name
        new_name = new_name + 1


if __name__ == '__main__':
    processor = TiffImageProcessor(r"D:\通用\image\通用变化.tif", r"D:\变化监测\label3", 512, 0.5)
    processor.tif_crop()