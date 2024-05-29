from osgeo import gdal
import numpy as np
import os

class ImageProcessor:
    def __init__(self):
        self.driver = gdal.GetDriverByName("GTiff")  # Create GDAL driver for GeoTIFF

    def read_img(self, input_file):
        in_ds = gdal.Open(input_file)
        rows = in_ds.RasterYSize  # Get data height
        cols = in_ds.RasterXSize  # Get data width
        bands = in_ds.RasterCount  # Get number of bands
        datatype = in_ds.GetRasterBand(1).DataType

        array_data = in_ds.ReadAsArray()  # Read data into array
        del in_ds

        return array_data, rows, cols, bands, datatype

    def write_img(self, read_path, img_array, img_transf, img_proj, output_file):
        im_width = img_array.shape[2]
        im_height = img_array.shape[1]
        img_bands = img_array.shape[0]

        dataset = self.driver.Create(output_file, im_width, im_height, img_bands, gdal.GDT_Byte)
        dataset.SetGeoTransform(img_transf)  # Write geotransform parameters
        dataset.SetProjection(img_proj)  # Write projection information
        for i in range(img_bands):
            dataset.GetRasterBand(i + 1).WriteArray(img_array[i])
        del dataset

    def compress(self, origin_16, output_file, low_per_raw=0.02, high_per_raw=0.98):
        array_data, rows, cols, bands, datatype = self.read_img(origin_16)

        compress_data = np.zeros((bands, rows, cols), dtype="uint8")

        for i in range(bands):
            cnt_array = np.where(array_data[i, :, :], 0, 1)
            num0 = np.sum(cnt_array)
            kk = num0 / (rows * cols)

            low_per = low_per_raw + kk - low_per_raw * kk
            low_per = low_per * 100
            high_per = (1 - high_per_raw) * (1 - kk)
            high_per = 100 - high_per * 100

            cutmin = np.percentile(array_data[i, :, :], low_per)
            cutmax = np.percentile(array_data[i, :, :], high_per)

            data_band = array_data[i]
            data_band[data_band < cutmin] = cutmin
            data_band[data_band > cutmax] = cutmax
            compress_data[i, :, :] = np.around((data_band[:, :] - cutmin) * 255 / (cutmax - cutmin))

        read_pre_dataset = gdal.Open(origin_16)
        img_transf = read_pre_dataset.GetGeoTransform()
        img_proj = read_pre_dataset.GetProjection()
        self.write_img(origin_16, compress_data, img_transf, img_proj, output_file)

if __name__ == '__main__':
    processor = ImageProcessor()
    input_dir = r"D:\变化监测\im2"

    for file_name in os.listdir(input_dir):
        input_file = os.path.join(input_dir, file_name)
        processor.compress(input_file, input_file)
