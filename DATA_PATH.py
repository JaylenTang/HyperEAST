Band4Depth4 = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]  # patch_size=5,window_size=25时，4层网络允许的光谱通道数
Band4Depth3 = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]    # patch_size=5,window_size=25时，3层网络允许的光谱通道数




# (145,145,200)，16
IN_PATH = r'hsi_data/IndianPine/indian_pines.mat'
IN_GT_PATH = r'hsi_data/IndianPine/indian_pines_gt.mat'
IP_LABELS = [            
            'Undefined',         # 0: 未定义类
            'Alfalfa',           # 1: 苜蓿
            'Corn-notill',       # 2: 玉米-免耕
            'Corn-mintill',      # 3: 玉米-少耕
            'Corn',              # 4: 玉米
            'Grass-pasture',     # 5: 草地-牧场
            'Grass-trees',       # 6: 草地-树木
            'Grass-pasture-mowed', # 7: 草地-修剪牧场
            'Hay-windrowed',     # 8: 干草-成行
            'Oats',              # 9: 燕麦
            'Soybean-notill',    # 10: 大豆-免耕
            'Soybean-mintill',   # 11: 大豆-少耕
            'Soybean-clean',     # 12: 大豆-净耕
            'Wheat',             # 13: 小麦
            'Woods',             # 14: 森林
            'Buildings-Grass-Trees-Drives', # 15: 建筑物-草地-树木-道路
            'Stone-Steel-Towers' # 16: 石头-钢塔
        ]

# (1217, 303, 274),16
WHU_Hi_HanChuan = r'hsi_data/WHU_Hi_HanChuan/WHU_Hi_HanChuan.mat'
WHU_Hi_HanChuan_GT = r'hsi_data/WHU_Hi_HanChuan/WHU_Hi_HanChuan_gt.mat'
HC_LABELS = [ 
            'Undefined',            # 未定义
            'Paddy rice',           # 水稻
            'Dryland crop',         # 旱地作物
            'Vegetables',           # 蔬菜
            'Cotton',               # 棉花
            'Orchard',              # 果园
            'Aquaculture water',    # 水产养殖用水
            'Residential area',     # 居住区
            'Industrial area',      # 工业区
            'Road',                 # 道路
            'Water',                # 水域
            'Woodland',             # 林地
            'Grassland',            # 草地
            'Bare land',            # 裸地
            'Railway',              # 铁路
            'Urban green space',    # 城市绿地
            'Other',                # 其他
]


# (550, 400, 270),9
WHU_Hi_LongKou = r'hsi_data/WHU_Hi_LongKou/WHU_Hi_LongKou.mat'
WHU_Hi_LongKou_GT = r'hsi_data/WHU_Hi_LongKou/WHU_Hi_LongKou_gt.mat'
LK_LABELS = [
            'Undefined',
            'Corn',
            'Cotton',
            'Sesame',
            'Broad-leaf soybean',
            'Narrow-leaf soybean',
            'Rice',
            'Water',
            'Roads and houses',
            'Mixed weed',
        ]