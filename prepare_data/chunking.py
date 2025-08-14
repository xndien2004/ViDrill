import re

def count_tokens(text):
    return len(text.split())

def get_last_overlap(text):
    section_pattern = r'\n*(\d{1,2}\..*?)(?=\n\d{1,2}\.|$)'
    sections = re.findall(section_pattern, text.strip(), flags=re.DOTALL)
    if sections:
        return sections[-1].strip()

    paras = re.split(r'\n\s*\n', text.strip())
    if len(paras) > 0:
        return paras[-1].strip()
    
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    if len(sentences) > 0:
        return sentences[-1].strip()
    
    return text.strip()

def split_chunks(text, max_len=2000):
    section_pattern = r'\n*(\d{1,2}\..*?)(?=\n\d{1,2}\.|$)'
    sections = re.findall(section_pattern, text, flags=re.DOTALL)

    if not sections:
        sections = [text.strip()]

    raw_chunks = []
    current_chunk_content = ""

    for section in sections:
        section = section.strip()
        
        tokens_to_add = count_tokens(section) + (1 if current_chunk_content else 0)

        if count_tokens(current_chunk_content) + tokens_to_add <= max_len:
            if current_chunk_content:
                current_chunk_content += "\n" + section
            else:
                current_chunk_content = section
        else:
            if current_chunk_content:
                raw_chunks.append(current_chunk_content)
                
            if count_tokens(section) > max_len:
                paras = re.split(r'\n\s*\n', section)
                sub_chunk_content = ""
                for para in paras:
                    para = para.strip()
                    para_tokens_to_add = count_tokens(para) + (2 if sub_chunk_content else 0)
                    
                    if count_tokens(sub_chunk_content) + para_tokens_to_add <= max_len:
                        if sub_chunk_content:
                            sub_chunk_content += "\n\n" + para
                        else:
                            sub_chunk_content = para
                    else:
                        if sub_chunk_content:
                            raw_chunks.append(sub_chunk_content)
                        sub_chunk_content = para
                
                if sub_chunk_content:
                    raw_chunks.append(sub_chunk_content)
                
                current_chunk_content = ""
            else:
                current_chunk_content = section

    if current_chunk_content:
        raw_chunks.append(current_chunk_content)

    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            final_chunks.append(chunk)
        else:
            overlap = get_last_overlap(raw_chunks[i-1])
            
            current_overlap = overlap

            while True:
                potential_new_chunk = (current_overlap + "\n\n" + chunk).strip()
                
                if count_tokens(potential_new_chunk) <= max_len:
                    final_chunks.append(potential_new_chunk)
                    break
                
                if len(current_overlap.split()) > 1:
                    current_overlap = ' '.join(current_overlap.split()[1:])
                elif len(current_overlap) > 0:
                    current_overlap = current_overlap[1:]
                else:
                    final_chunks.append(chunk)
                    break

    return final_chunks
