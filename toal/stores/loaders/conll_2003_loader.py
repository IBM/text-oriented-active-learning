import pandas as pd
import random

# TODO: to rewrite as a loader, this is the old code


def load_conll_2003(csv_path, split_type, include_sentences_no_entity=True, default_label='O'):
    annotation_unit_counter = -1
    doc_start = False
    sentences = []
    labels = []
    splits = []
    ann_unit_id = []
    with open(csv_path) as f:
        sentence = []
        label_sent = []
        for line in f:
            if line.startswith("-DOCSTART-"):
                annotation_unit_counter += 1
                doc_start = True
            elif line.strip() == "":  # empty lines break sentences
                if not doc_start:  # but ignore those following DOCSTART
                    if len(sentence) > 0:
                        # optionally add only sentences with some label
                        if include_sentences_no_entity or any([label != default_label for label in label_sent]):
                            sentences.append(' '.join(sentence))
                            labels.append(' '.join(label_sent))
                            # splits.append(split_i == 0 and 'train' or split_i == 1 and 'test' or 'unlabeled')
                            ann_unit_id.append(str(annotation_unit_counter))
                    sentence = []
                    label_sent = []
                doc_start = False
            else:
                doc_start = False
                cols = line.split()
                sentence.append(cols[0])  # add word
                label_sent.append(cols[3])  # add label
        # make sure last sentence is added
        if len(sentence) > 0:
            # optionally add only sentences with some label
            if include_sentences_no_entity or any([label != default_label for label in label_sent]):
                sentences.append(' '.join(sentence))
                labels.append(' '.join(label_sent))
                # splits.append(split_i == 0 and 'train' or split_i == 1 and 'test' or 'unlabeled')
                ann_unit_id.append(str(annotation_unit_counter))

    unlabeled_df = None
    labeled_df = None

    if split_type == 'unlabeled':
        unlabeled_df = pd.DataFrame(sentences, columns=['text'])
        unlabeled_df['annotation_unit_id'] = None
    else:
        # TODO Chances are this can be optimised, list+zip sound memory expensive
        labeled_df = pd.DataFrame(list(zip(sentences, labels)), columns=['text', 'label'])
        labeled_df['train'] = split_type == "train"
        labeled_df['annotation_unit_id'] = None

    return labeled_df, unlabeled_df


def load_split_conll_2003(csv_path, include_sentences_no_entity=True, default_label='O', test_pct=0.2,
                          test_n=None, labeled_pct=0.2, labeled_n=None, shuffle=False):
    annotation_unit_counter = -1
    doc_start = False
    sentences = []
    labels = []
    splits = []
    ann_unit_id = []
    with open(csv_path) as f:
        sentence = []
        label_sent = []
        for line in f:
            if line.startswith("-DOCSTART-"):
                annotation_unit_counter += 1
                doc_start = True
            elif line.strip() == "":  # empty lines break sentences
                if not doc_start:  # but ignore those following DOCSTART
                    if len(sentence) > 0:
                        # optionally add only sentences with some label
                        if include_sentences_no_entity or any([label != default_label for label in label_sent]):
                            sentences.append(' '.join(sentence))
                            labels.append(' '.join(label_sent))
                            # splits.append(split_i == 0 and 'train' or split_i == 1 and 'test' or 'unlabeled')
                            ann_unit_id.append(str(annotation_unit_counter))
                    sentence = []
                    label_sent = []
                doc_start = False
            else:
                doc_start = False
                cols = line.split()
                sentence.append(cols[0])  # add word
                label_sent.append(cols[3])  # add label
        # make sure last sentence is added
        if len(sentence) > 0:
            # optionally add only sentences with some label
            if include_sentences_no_entity or any([label != default_label for label in label_sent]):
                sentences.append(' '.join(sentence))
                labels.append(' '.join(label_sent))
                # splits.append(split_i == 0 and 'train' or split_i == 1 and 'test' or 'unlabeled')
                ann_unit_id.append(str(annotation_unit_counter))


    # TODO Chances are this can be optimised, list+zip sound memory expensive
    rows = list(zip(sentences, labels))
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
