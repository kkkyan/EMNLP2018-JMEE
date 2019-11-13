import json
from pytorch_pretrained_bert import BertTokenizer


def main():
    data_path = "./ace-05-splits/train.json"
    save_file = "./pretrained_text.txt"
    split_file = "./split_words.txt"
    
    tokenizer = BertTokenizer.from_pretrained("/home/yk/.pytorch_pretrained_bert/bert-base-uncased",
                                              never_split=["[CLS]", "[SEP]"])
    with open(data_path, "r") as f, open(save_file, "w") as fw, open(split_file, "w") as sw:
        lines = json.load(f)
        e_count = 0
        miss = 0
        bert_miss = 0
        for l in lines:
            sent = l["sentence"]
            fw.write(sent)
            fw.write("\n")
            
            events = l["golden-event-mentions"]
            if len(events) == 0:
                continue
            
            e_count += 1
            sent_split = sent.split(" ")
            for e in events:
                trigger = e["trigger"]["text"]
                if trigger not in sent_split:
                    miss += 1
                    
                    sw.write(sent + "\n")
                    sw.write(trigger + "\n")
                    tokens = tokenizer.tokenize(sent)
                    tokens_str = " ".join(tokens)
                    sw.write(tokens_str + "\n")
                    
                    if trigger not in tokens:
                        bert_miss += 1
                        sw.write("BERT_TOKEN_MISS\n")

                    sw.write("\n")
            
    print("events:", e_count)
    print("miss:", miss, " rate:", miss/e_count * 100)
    print("bert token miss:", bert_miss, " rate:", bert_miss/e_count * 100)


if __name__ == '__main__':
    main()
