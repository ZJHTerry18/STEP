from src.tokenizer import MultimodalLlamaTokenizer

tokenizer = MultimodalLlamaTokenizer.from_pretrained(
    pretrained_model_name_or_path="BAAI/Emu2",
    local_files_only=True)

text = ""

toregressed = tokenizer.build_input_ids(
                    text=[text],
                    max_length=1000000,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_tensors='pt'
                )

print(toregressed['input_ids'][0].dtype)

print(toregressed['input_ids'][0])

print(toregressed['input_ids'])