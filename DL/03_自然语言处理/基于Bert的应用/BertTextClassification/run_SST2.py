# coding=utf-8
from main import main


if __name__ == "__main__":

    model_name = "BertLSTM"
    label_list = ['0', '1']
    data_dir = "/search/hadoop02/suanfa/songyingxin/SongWork/PaperDataset/sst_2"
    output_dir = ".sst_output/" 
    cache_dir = ".sst_cache/"
    log_dir = ".sst_log/" 

    # bert-base
    bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-uncased-vocab.txt"
    bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-uncased"

    # # bert-large
    # bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-large-uncased-vocab.txt"
    # bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-large-uncased"

    if model_name == "BertOrigin":
        from BertOrigin import args

    elif model_name == "BertCNN":
        from BertCNN import args

    elif model_name == 'BertLSTM':
        from BertLSTM import args

    elif model_name == "BertATT":
        from BertATT import args

    elif model_name == "BertRCNN":
        from BertRCNN import args

    elif model_name == "BertCNNPlus":
        from BertCNNPlus import args
    
    elif model_name == "BertDPCNN":
        from BertDPCNN import args

    config = args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir)

    main(config, config.save_name, label_list)
        

