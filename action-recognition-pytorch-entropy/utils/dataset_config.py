

DATASET_CONFIG = {
    'SBU':{
        'num_classes_target': 8,
        'num_classes_budget': 13,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': 'rgb_{:06d}.png',
        'filter_video': 3
    },
    'KTH':{
        'num_classes_target': 6,
        'num_classes_budget': 25,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:04d}.png',
        'filter_video': 3
    },
    'IPN':{
        'num_classes_target': 14,
        'num_classes_budget': 2,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:07d}.jpg',
        'filter_video': 3
    },
    'st2stv2': {
        'num_classes': 174,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 3
    },
    'mini_st2stv2': {
        'num_classes': 87,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'test_list_name': 'mini_test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 3
    },
    'kinetics400': {
        'num_classes': 8,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 4
    },
    'mini_kinetics400': {
        'num_classes': 200,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'test_list_name': 'mini_test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'moments': {
        'num_classes': 339,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'mini_moments': {
        'num_classes': 200,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
}


def get_dataset_config(dataset):
    ret = DATASET_CONFIG[dataset] 
    num_classes_target = ret['num_classes_target']
    num_classes_budget = ret['num_classes_budget']
    # num_classes = ret['num_classes']
    train_list_name = ret['train_list_name']
    val_list_name = ret['val_list_name']
    test_list_name = ret.get('test_list_name', None)
    filename_seperator = ret['filename_seperator']
    image_tmpl = ret['image_tmpl']
    filter_video = ret.get('filter_video', 0)
    label_file = ret.get('label_file', None)

    return num_classes_target, num_classes_budget, train_list_name, val_list_name, test_list_name, filename_seperator, \
           image_tmpl, filter_video, label_file
