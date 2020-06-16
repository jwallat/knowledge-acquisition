from transformers import BertConfig, AutoTokenizer, BertModel
from knowledge_probing.models.lightning.decoder import Decoder
from knowledge_probing.models.lightning.hugging_decoder import HuggingDecoder


def get_model(args):
    # Get config for Decoder
    config = BertConfig.from_pretrained(args.bert_model_type)
    # if args.probing_layer != 12:
    config.output_hidden_states = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_model_type, use_fast=False)

    # Load Bert as BertModel which is plain and has no head on top
    if args.bert_type == 'default':
        bert = BertModel.from_pretrained(args.bert_model_type, config=config)
    elif args.bert_type == 'qa':
        bert = BertModel.from_pretrained(args.qa_model_dir, config=config)

    # Make sure the bert model is not trained
    bert.eval()
    bert.requires_grad = False
    for param in bert.parameters():
        param.requires_grad = False
    bert.to(args.device)

    # Get the right decoder
    if args.training_decoder == "Decoder":
        decoder = Decoder(hparams=args, bert=bert, config=config)
    else:
        decoder = HuggingDecoder(hparams=args, bert=bert, config=config)

    return decoder
