import math
import requests
import os
import cv2
import gdal
from osgeo import osr
import numpy as np

class MapDownloader:
    def __init__(self, url, min_lat, max_lat, min_lon, max_lon, zoom_level):
        self.url = url
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.zoom_level = zoom_level

    def deg2num3857(self, lon, lat):
        # implementation of deg2num3857
        lat_rad = math.radians(lat)
        n = 2.0 ** self.zoom_level
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    def num2deg3857(self, x, y):
        # implementation of num2deg3857
        n = 2.0 ** self.zoom_level
        lon_min = x / n * 360.0 - 180.0
        rad_min = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
        lat_min = math.degrees(rad_min)
        lon_max = ((x + 1) / n) * 360.0 - 180.0
        rad_max = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_max = math.degrees(rad_max)
        return lon_min, lat_min, lon_max, lat_max

    def download_tiles(self):
        # implementation of download_tiles
        min_x, max_y = self.deg2num3857(self.min_lon, self.min_lat)
        max_x, min_y = self.deg2num3857(self.max_lon, self.max_lat)

        result = np.zeros((256 * (max_y - min_y + 1), 256 * (max_x - min_x + 1), 3), dtype=np.uint8)
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tile_url = self.url.format(x=x, y=y, z=self.zoom_level)
                response = requests.get(tile_url)
                if response.status_code == 200:
                    nparr = np.frombuffer(response.content, np.uint8)
                    tile = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    tile[:, :, 0], tile[:, :, 2] = tile[:, :, 2], tile[:, :, 0].copy()
                    result[256 * (y - min_y):256 * (y - min_y + 1), 256 * (x - min_x):256 * (x - min_x + 1)] = tile
        return result

    def create_georeferenced_tiff(self, image, left_bottom_x, left_bottom_y, right_top_x, right_top_y, output_file):
        # implementation of create_georeferenced_tiff
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        transform = [left_bottom_x, (right_top_x - left_bottom_x) / image.shape[1], 0, right_top_y, 0, -(right_top_y - left_bottom_y) / image.shape[0]]

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_file, image.shape[1], image.shape[0], 3, gdal.GDT_Byte)

        dataset.SetGeoTransform(transform)
        dataset.SetProjection(srs.ExportToWkt())

        for i in range(3):
            band = dataset.GetRasterBand(i + 1)
            band.SetNoDataValue(0)
            band.WriteArray(image[:,:,i])

        dataset.FlushCache()
        
if __name__ == '__main__':
    # 设置地图服务链接和地理范围
    url = 'https://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    min_lon,max_lat = 72.189244  ,55.577454
    max_lon,min_lat = 136.220539  ,20.813529 
    zoom_level = 8

    # 创建地图下载器实例并使用它下载地图瓦片并创建地理参考的GeoTIFF文件
    map_downloader = MapDownloader(url, min_lat, max_lat, min_lon, max_lon, zoom_level)
    result_image = map_downloader.download_tiles()
    lon_min, lat_min, lon_max, lat_max = map_downloader.num2deg3857(*map_downloader.deg2num3857(min_lon, min_lat))
    lon_min1, lat_min1, lon_max1, lat_max1 = map_downloader.num2deg3857(*map_downloader.deg2num3857(max_lon, max_lat))
    map_downloader.create_georeferenced_tiff(result_image, lon_min, lat_min, lon_max1, lat_max1, "D:/image/output_image.tif")
