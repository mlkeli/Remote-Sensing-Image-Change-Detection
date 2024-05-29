data_root = 'D:/jianzu'
batch_size = 2
val_batch_size = 2
test_batch_size = 1
epochs = 30
lr = 0.01
weight_decay = 1e-4
backbone = 'hrnet_w18'
model = 'pspnet'
lightweight = False
pretrain_from = None
load_from = None
pretrained = False
tta = False
save_mask = False
use_pseudo_label = False



# 训练
epochs = 20

# 预测

# 星图地球
# url = 'https://tiles1.geovisearth.com/base/v1/img/{z}/{x}/{y}?format=webp&tmsIds=w&token=0aeb02f29320b060c2e2d0c04eb4887c6b8d5a8ed479b3aacff2b6a273b0d38d'

# esri影像
url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/18820/{z}/{y}/{x}'
url_after = 'https://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'

# 天地图影像
# url = 'https://t3.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TileMatrix={z}&TileRow={y}&TileCol={x}&tk=4267820f43926eaf808d61dc07269beb'

# 高德影像
# url = 'http://wprd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=6&x={x}&y={y}&z={z}'
min_lon,max_lat = 108.674752  ,34.279606
max_lon,min_lat = 108.694752  ,34.259606
zoom_level = 17
result = r'D:\image\ceshi'
image = r"D:\image\tif4.tif"
image_after = r"D:\image\tif4_after.tif"
