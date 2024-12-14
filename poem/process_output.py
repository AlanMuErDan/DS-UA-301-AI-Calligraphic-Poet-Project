k = 32

input_file = "./results/poem-top-"+str(k)+"/poem-out.txt"
output_file = "./results/poem-top-"+str(k)+"/processed_poem.txt"

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    count = 0  # 用于计数每写入两个生成的诗句后换行
    for line in infile:
        # 跳过空行
        if not line.strip():
            continue
        
        # 尝试分割行，若格式不正确，跳过该行
        try:
            input_text, generated_text = line.strip().split("\t")
        except ValueError:
            print(f"Skipping malformed line: {line.strip()}")
            continue
        
        # 处理生成的文本：去掉特殊标签和换行符
        generated_text = generated_text.replace("<bos>", "").replace("<eos>", "").replace("</s>", "").strip()
        
        # 写入生成的文本
        outfile.write(generated_text + " ")
        count += 1
        
        # 每两个生成的诗句换一次行
        if count % 2 == 0:
            outfile.write("\n")

print("Processing complete!")
