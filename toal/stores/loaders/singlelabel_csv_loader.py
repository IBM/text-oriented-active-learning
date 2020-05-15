import pandas as pd


def _csv_to_df(csv_path, headers):
    """Load a training set from a csv"""

    # Assume all columns are strings
    columns_types = {i: str for i, header in enumerate(headers)}

    temp_df = pd.read_csv(csv_path, converters=columns_types, skip_blank_lines=False)
    # TODO: check that there are only two columns of type string, then convert to our format
    temp_df.columns = headers
    # Add the column split, this is all training data
    temp_df['annotation_unit_id'] = None
    return temp_df


def load_single_label_csv(csv_path, split_type):

    if split_type not in ["train", "test", "unlabeled"]:
        raise ValueError(f"split_type must be one of 'train', 'test', or 'unlabeled', received {split_type}")

    unlabeled_df = None
    labeled_df = None

    if split_type == 'train' or split_type == 'test':
        labeled_df = _csv_to_df(csv_path, headers=['text', 'label'])
        is_train_data = split_type == "train"
        labeled_df['train'] = is_train_data
    else:
        unlabeled_df = _csv_to_df(csv_path, headers=['text'])

    return labeled_df, unlabeled_df
