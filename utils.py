import os


def load_all_poems(src_dir):

    all_poems = []
    
    poet_cnt = 0
    poem_cnt = 0
    
    for poet in os.listdir(src_dir):
        poet_cnt += 1
        
        if not os.path.isdir(os.path.join(src_dir, poet)):
            print(f"pass {poet}")
            continue
        
        poet_chn_name, _ = poet.split("_")[:2]
        
        for poetry in os.listdir(os.path.join(src_dir, poet)):
            
            if not poetry.endswith("pt"):
                continue
                
            poem = {"author": poet_chn_name}

            start_parsing = False

            with open(os.path.join(src_dir, poet, poetry)) as frd:
                all_lines = []
                
                for line in frd.readlines():

                    if not start_parsing:
                        if line.startswith("title"):
                            title = line.strip().split(":")[-1]
                            poem["title"] = title
                        elif line.startswith("date"):
                            date = line.strip().split(":")[-1]
                            if not date:
                                poem["date"] = date if date else ""
                            start_parsing = True
                    else:
                        line = line.strip()
                        
                        if len(line) == 0:
                            if all_lines:
                                all_lines.append("")
                        else:
                            all_lines.append(line)
                
                poem["body"] = all_lines

            all_poems.append(poem)
            poem_cnt += 1
    
    print(f"Done parsing, total poet:{poet_cnt}, total poem:{poem_cnt}")

    return all_poems


def build_plain_text_from_all_poems(all_poems, export_to):
    
    with open(export_to, "w") as fwt:
        for poem in all_poems:
            title = list(poem["title"])
            body = poem["body"]
            
            fwt.write(" ".join(title))
            fwt.write("\n")
            for line in body:
                fwt.write(" ".join(list(line)))
                fwt.write("\n")
            fwt.write("\n")
                
    print("Done export to plain text")


def translate_logits(logits, idx_to_str, unk, eos):
    """Translates logits to tokens.

    Args:
        logits (Tensor): logit matrix, bs, max_seq_len
        idx_to_str (dict): mapper from idx to str
    """
    bs, max_seq_len = logits.size()
    logits = logits.cpu().numpy().tolist()
    
    results = []
    
    for batch in range(bs):
        curr_batch_result = []
        for idx in logits[batch]:
            if idx == eos:
                break
            else:
                curr_batch_result.append(idx_to_str.get(idx, unk))
        results.append(curr_batch_result)
        
    return results
