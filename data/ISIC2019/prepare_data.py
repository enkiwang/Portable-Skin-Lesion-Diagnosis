import sys
import os
from loader import parse_metadata
from loader import split_k_folder_csv, label_categorical_to_number
from loader import split_train_test_imgs_csv, merge_labels_and_metadata


SEED = 2021
ISIC_BASE_PATH = "./" 
CSV_META_PATH = "ISIC_2019_Training_GroundTruth.csv"
CSV_LABELS_PATH = "ISIC_2019_Training_Metadata.csv"
IMGS_BASE_DIR= 'ISIC_2019_Training_Input/'
TRAIN_DIR_BASE = 'train/'
TEST_DIR_BASE = 'test/'
TRAIN_RATIO = 0.9
TEST_RATIO = 0.1
FOLD_NUM = 5
TRAIN_CSV_NAME = 'ISIC2019_parsed_train.csv'
TEST_CSV_NAME = 'ISIC2019_parsed_test.csv'

## merge csv files
df_merge = merge_labels_and_metadata(CSV_META_PATH, CSV_LABELS_PATH)

## parse metadata
data = parse_metadata (df_merge, replace_nan="missing",
           cols_to_parse=['sex', 'anatom_site_general'], replace_rules={"age_approx": {"missing": 0}})


## save train/test csv and image files
TRAIN_DIR = TRAIN_DIR_BASE + 'imgs/'
TEST_DIR = TEST_DIR_BASE + 'imgs/'
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

PATH_TRAIN_CSV = TRAIN_DIR_BASE + TRAIN_CSV_NAME
PATH_TEST_CSV = TEST_DIR_BASE + TEST_CSV_NAME

split_train_test_imgs_csv (data, save_path_train_csv=PATH_TRAIN_CSV, \
                            save_path_test_csv=PATH_TEST_CSV, \
                            save_imgs_test_dir=TEST_DIR, save_imgs_train_dir=TRAIN_DIR, \
                            imgs_base_dir=IMGS_BASE_DIR, \
                            tr=TRAIN_RATIO, te=TEST_RATIO, seed_number=SEED)  

## split training data as train and validation
data_train_cv = split_k_folder_csv(PATH_TRAIN_CSV, "diagnostic", k_folder=FOLD_NUM, seed_number=SEED)

## label to categorical
data_train = label_categorical_to_number (data_train_cv, "diagnostic", col_target_number="diagnostic_number")
data_train.to_csv(PATH_TRAIN_CSV, index=False)  

data_test = label_categorical_to_number (PATH_TEST_CSV, "diagnostic", col_target_number="diagnostic_number")
data_test.to_csv(PATH_TEST_CSV, index=False)





















