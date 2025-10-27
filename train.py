from ultralytics import YOLO
import yaml
import os

def update_dataset_yaml_path(relative_yaml_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file_path = os.path.join(script_dir, relative_yaml_path)
    yaml_file_path = os.path.normpath(yaml_file_path)
    if not os.path.exists(yaml_file_path):
        print(f"Error: yaml not exits: {yaml_file_path}")
        return
    new_path_value = os.path.dirname(yaml_file_path)
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    data['Path'] = new_path_value
    with open(yaml_file_path, 'w') as file:
        yaml.dump(data, file, sort_keys=False)


script_path = os.path.abspath(__file__)
dir_path = os.path.dirname(script_path)

# model_cfgs: yolov5n.yaml  yolov8n.yaml  yolo11.yaml  yolo12.yaml  rtdetr-resnet50.ymal
# model_config = "yolov8n.yaml"
model_config = "yolo11n-seg.yaml"
model = YOLO(model_config)
# dataset = dir_path + "/../ACNEDet v1.v3-ul-dataset/ACNEDet v1.v3.yaml"
# dataset = dir_path + "/../ACNEDet v1.v3-yoloseg-dataset/ACNEDet v1.v3.yaml"
dataset = r"D:\skinpilot\SkinDetect\SkinDetect\hairline_data\data.yaml"
# update_dataset_yaml_path(dataset)

if __name__ == '__main__':
    model.train(data=dataset, epochs=200, device="cuda:0", pretrained=False, batch=8)