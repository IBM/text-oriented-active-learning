import pandas as pd
import json


def load_w_annotation_units_from_watson_discover(wds_resp, enrichment='relations', filter_type=None):

    # we are only looking at unlabeled data.  labeled data is handled in WKS
    # can I have a None labeled dataframe?
    labeled_df = None

    # 1) load the json from a file -> it's gonna be replaced with a rest call

    # 2) Parse the json to have the desired dataframes -> this part is gonna be kept
    # TODO better/safer way to read JSON (that isn't too verbose)?
    # texts = ["%s;;;Type: %s;;;Score: %s;;;%s:%s---%s:%s" % (relation['sentence'], relation['type'], relation['score'],
    #                                             relation['arguments'][0]['entities'][0]['text'],
    #                                             relation['arguments'][0]['entities'][0]['type'],
    #                                             relation['arguments'][1]['entities'][0]['text'],
    #                                             relation['arguments'][1]['entities'][0]['type'])
    if enrichment == 'entities':
        raise NotImplementedError
    else:
        expanded_texts = [selected_json_string(relation['sentence'], relation['type'], relation['score'],
                                               relation['arguments'][0]['entities'][0]['text'],
                                               relation['arguments'][0]['entities'][0]['type'],
                                               relation['arguments'][1]['entities'][0]['text'],
                                               relation['arguments'][1]['entities'][0]['type'])
                          for result in wds_resp['results']
                          for relation in result['enriched_text']['relations'] if filter_type is None or filter_type == relation['type']]
        texts = [relation['sentence'] for result in wds_resp['results']
                 for relation in result['enriched_text']['relations'] if filter_type is None or filter_type == relation['type']]

    rows = list(zip(expanded_texts, texts))
    unlabeled_df = pd.DataFrame(rows, columns=['text', 'annotation_unit_id'])

    # 3) return the dataframe
    return labeled_df, unlabeled_df


def load_from_watson_discover(wds_resp, enrichment='relations', filter_type=None):

    # we are only looking at unlabeled data.  labeled data is handled in WKS
    # can I have a None labeled dataframe?
    labeled_df = None

    # 1) load the json from a file -> it's gonna be replaced with a rest call

    # 2) Parse the json to have the desired dataframes -> this part is gonna be kept
    # TODO better/safer way to read JSON (that isn't too verbose)?
    # texts = ["%s;;;Type: %s;;;Score: %s;;;%s:%s---%s:%s" % (relation['sentence'], relation['type'], relation['score'],
    #                                             relation['arguments'][0]['entities'][0]['text'],
    #                                             relation['arguments'][0]['entities'][0]['type'],
    #                                             relation['arguments'][1]['entities'][0]['text'],
    #                                             relation['arguments'][1]['entities'][0]['type'])
    if enrichment == 'entities':
        texts = [selected_entities_json_string(relation['text'], relation['type'], relation['confidence'])
                 for result in wds_resp['results']
                 for relation in result['enriched_text']['entities'] if filter_type is None or filter_type == relation['type']]

    else:
        texts = [selected_json_string(relation['sentence'], relation['type'], relation['score'],
                                                           relation['arguments'][0]['entities'][0]['text'],
                                                           relation['arguments'][0]['entities'][0]['type'],
                                                           relation['arguments'][1]['entities'][0]['text'],
                                                           relation['arguments'][1]['entities'][0]['type'])
                 for result in wds_resp['results']
                 for relation in result['enriched_text']['relations'] if filter_type is None or filter_type == relation['type']]

    unlabeled_df = pd.DataFrame(texts, columns=['text'])
    unlabeled_df['annotation_unit_id'] = None

    # 3) return the dataframe
    return labeled_df, unlabeled_df


def selected_json_string(sentence, type, score, arg0_text, arg0_type, arg1_text, arg1_type):
    d = {'sentence': sentence, 'type': type, 'score': score, 'arg0_text': arg0_text, 'arg0_type': arg0_type,
         'arg1_text': arg1_text, 'arg1_type': arg1_type};
    return json.dumps(d)


def selected_entities_json_string(text, type, confidence):
    d = {'text': text, 'type': type, 'confidence': confidence};
    return json.dumps(d)
