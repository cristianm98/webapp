from collections import OrderedDict

camvid_encoding = OrderedDict([
    ('Animal', (64, 128, 64)),
    ('Archway', (192, 0, 128)),
    ('Bicyclist', (0, 128, 92)),
    ('Bridge', (0, 128, 64)),
    ('Building', (128, 0, 0)),
    ('Car', (64, 0, 128)),
    ('CartLuggagePram', (64, 0, 192)),
    ('Child', (192, 128, 64)),
    ('Column_Pole', (192, 192, 128)),
    ('Fence', (64, 64, 128)),
    ('LaneMkgsDriv', (128, 0, 192)),
    ('LaneMkgsNonDriv', (192, 0, 64)),
    ('Misc_Text', (128, 128, 64)),
    ('MotorcycleScooter', (192, 0, 192)),
    ('OtherMoving', (128, 64, 64)),
    ('ParkingBlock', (64, 192, 128)),
    ('Pedestrian', (64, 64, 0)),
    ('Road', (128, 64, 128)),
    ('RoadShoulder', (128, 128, 192)),
    ('Sidewalk', (0, 0, 192)),
    ('SignSymbol', (192, 128, 128)),
    ('Sky', (128, 128, 128)),
    ('SUVPickupTruck', (64, 128, 192)),
    ('TrafficCone', (0, 0, 64)),
    ('TrafficLight', (0, 64, 64)),
    ('Train', (192, 64, 128)),
    ('Tree', (128, 128, 0)),
    ('Truck_Bus', (192, 128, 192)),
    ('Tunnel', (64, 0, 64)),
    ('VegetationMisc', (192, 192, 0)),
    ('Void', (0, 0, 0)),
    ('Wall', (64, 192, 0))
])

infrared_encoding = OrderedDict([
    ('Background', (0, 0, 0)),
    ('Road', (128, 0, 0))
])


def get_color_encoding(dataset_name):
    if dataset_name.lower() == 'camvid':
        return camvid_encoding
    elif dataset_name.lower() == 'infrared':
        return infrared_encoding
