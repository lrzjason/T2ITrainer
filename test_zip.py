import json

datarows = []
metadata_path = 'F:/ImageSet/kolors_slider/metadata_kolors_slider.json'
with open(metadata_path, "r", encoding='utf-8') as readfile:
    metadata = json.loads(readfile.read())
    generation_configs = metadata['generation_configs']
    pos_config = generation_configs[0]
    neg_config = generation_configs[1]
    uncondition_config = generation_configs[2]
    if len(pos_config['item_list']) != len(neg_config['item_list']):
        raise ValueError("Positive and Negative images must have same number of images")
    for pos_item, neg_item in zip(pos_config['item_list'], neg_config['item_list']):
        # datarows.append()
        print('pos_item')
        print(pos_item)
        print('neg_config')
        print(neg_config)
        break