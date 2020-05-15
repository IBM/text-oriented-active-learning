import pandas as pd
from csv import reader
import random


def _parse_multiclass_csv(csv_path, has_header=False):
    with open(csv_path, encoding='latin1') as f:
        csv_reader = reader(f)
        for line_i, line in enumerate(csv_reader):
            if has_header and line_i == 0:
                continue
            # texts.append(line[0])
            text = line[0]
            label = None
            if len(line) >= 2:  # we have a label
                label = ' '.join([l.replace(' ', '_') for l in line[1:]]).strip()
            yield text, label, line_i


def load_multi_label_csv(csv_path, split_type, has_header=False):
    """Load a training set from a multi label csv file"""

    if split_type not in ["train", "test", "unlabeled"]:
        raise ValueError(f"split_type must be one of 'train', 'test', or 'unlabeled', received {split_type}")

    texts = []
    labels = []

    for text, joint_labels, row_number in _parse_multiclass_csv(csv_path, has_header):

        if split_type == 'unlabeled':
            if joint_labels is not None:
                raise ValueError(f"Unlabeled test does not have labels, found '{joint_labels}' at line {str(row_number)}")
            texts.append(text)

        else:
            if joint_labels is None:
                raise ValueError(f"Train or test set must have labels, label not found at line {str(row_number)}")
            else:
                texts.append(text)
                labels.append(joint_labels)

    unlabeled_df = None
    labeled_df = None

    if split_type == 'unlabeled':
        unlabeled_df = pd.DataFrame(texts, columns=['text'])
        unlabeled_df['annotation_unit_id'] = None
        # temp_df['label'] = None
    else:
        # TODO Chances are this can be optimised, list+zip sound memory expensive
        labeled_df = pd.DataFrame(list(zip(texts, labels)), columns=['text', 'label'])
        labeled_df['train'] = split_type == "train"
        labeled_df['annotation_unit_id'] = None

    return labeled_df, unlabeled_df


def load_split_multi_label_csv(csv_path, has_header=False, test_pct=0.2, test_n=None, labeled_pct=0.2, labeled_n=None, shuffle=False):
    """Load from a multi label csv file and split into train/test and unlabeled splits"""

    texts = []
    labels = []

    for text, joint_labels, row_number in _parse_multiclass_csv(csv_path, has_header):
        if joint_labels is None:
            raise ValueError(f"Train or test set must have labels, label not found at line {str(row_number)}")
        else:
            texts.append(text)
            labels.append(joint_labels)

    # TODO Chances are this can be optimised, list+zip sound memory expensive
    rows = list(zip(texts, labels))
    if shuffle:
        random.seed(123)
        random.shuffle(rows)

    if test_n is not None and labeled_n is not None and test_n + labeled_n > len(rows):
        print("No data is left 'unlabeled'.")

    if test_n is None:
        test_n = int(test_pct * len(rows))
    if labeled_n is None:
        labeled_n = int(labeled_pct * len(rows))

    num_train_test = test_n + labeled_n
    labeled_df = pd.DataFrame(rows[:num_train_test], columns=['text', 'label'])
    labeled_df['train'] = [i >= test_n for i in range(num_train_test)]
    labeled_df['annotation_unit_id'] = None

    # key: text, value: label
    unlabeled_map = {t: l for t, l in rows[num_train_test:]}
    unlabeled_df = pd.DataFrame(list(unlabeled_map.keys()), columns=['text'])
    unlabeled_df['annotation_unit_id'] = None

    return labeled_df, unlabeled_df, unlabeled_map
