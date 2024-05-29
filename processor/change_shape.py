import cv2
import numpy as np
from osgeo import gdal, osr, ogr


def change_shape(input_raster):
    # 读取栅格数据
    raster = gdal.Open(input_raster)
    band = raster.GetRasterBand(1)
    data = band.ReadAsArray()

    # 转换为面
    _, binary = cv2.threshold(data, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建面数据集
    driver = ogr.GetDriverByName('ESRI Shapefile')
    output_shapefile = input_raster[:-4] + ".shp"
    out_ds = driver.CreateDataSource(output_shapefile)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(raster.GetProjection())

    out_layer = out_ds.CreateLayer('polygon', srs)
    fd = ogr.FieldDefn('value', ogr.OFTInteger)
    out_layer.CreateField(fd)

    for contour in contours:
        defn = out_layer.GetLayerDefn()
        feature = ogr.Feature(defn)
        poly = ogr.Geometry(ogr.wkbPolygon)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        
        for point in contour:
            x, y = point[0]
            x_geo = raster.GetGeoTransform()[0] + x * raster.GetGeoTransform()[1]
            y_geo = raster.GetGeoTransform()[3] + y * raster.GetGeoTransform()[5]
            ring.AddPoint(x_geo, y_geo)
        
        # Ensure the first and last points are the same to create a closed polygon
        ring.AddPoint(ring.GetX(0), ring.GetY(0))
        
        poly.AddGeometry(ring)
        feature.SetGeometry(poly)
        feature.SetField('value', 1)
        out_layer.CreateFeature(feature)
        
    # 关闭数据集
    out_ds = None
    


if __name__ == '__main__':
    change_shape(r"D:\change_shape\tif4.tif")
