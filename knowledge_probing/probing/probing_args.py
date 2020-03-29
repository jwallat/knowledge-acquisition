from knowledge_probing.file_utils import load_file


def build_args(dataset_name, lowercase):
    relations, data_path_pre, data_path_post = '', '', ''
    if dataset_name == 'Google_RE':
        relations, data_path_pre, data_path_post = get_GoogleRE_parameters()
    elif dataset_name == 'TREx':
        relations, data_path_pre, data_path_post = get_TREx_parameters()
    elif dataset_name == 'ConceptNet':
        relations, data_path_pre, data_path_post = get_ConceptNet_parameters()
    elif dataset_name == 'Squad':
        relations, data_path_pre, data_path_post = get_Squad_parameters()
    else:
        print('Could not find dataset in supported datasets: {}'.format(dataset_name))
        return

    relation_args = []  # Array for args for each run that has to be done
    for relation in relations:
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "template": "",
            "relation": relation['relation'],
            # "model_name": 'default', #QA
            # "bert_vocab_name": "vocab.txt",
            # "logdir": "output",
            # "lowercase": lowercase, # Have lowercase being handled purely by the given model and tokenizer
            "precision_at_k": 100,
            # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]

        relation_args.append(PARAMETERS)

    return relation_args


def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters():
    relations = [
        {
            "relation": "place_of_birth",
            "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },
        {
            "relation": "date_of_birth",
            "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    data_path_pre = "data/Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_ConceptNet_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "ConceptNet/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_Squad_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "Squad/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post
