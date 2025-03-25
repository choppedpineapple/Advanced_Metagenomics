import os
from multiprocessing import Pool, Manager
import time
def calculate_con_total(seq_list,final_header_list):
    con_total = []
    for p in seq_list:
        consensus_sequence = ""
        BASES_CALCULATE = {"A": 0, "C": 0, "G": 0, "T": 0, "N": 0}
        bases = ["A", "C", "G", "T", "N"]
        for k in range(len(p)):
            for base in bases:
                BASES_CALCULATE[base] = 0
            for j in range(len(seq_list)):
                if len(seq_list[j]) > k:
                    base = seq_list[j][k]
                    BASES_CALCULATE[base] += 1
            most_common_base = max(BASES_CALCULATE, key=BASES_CALCULATE.get)
            consensus_sequence += most_common_base
        for header in final_header_list:
            print(header)
            con_total.append(f'>{header}\n{consensus_sequence}')
    
            return con_total

def process_file(input_file_path):
    
    with open(input_file_path, 'r') as file_1:
        seq_list = []
        seq_count = []
        header_list = []
        final_header_list = []
        for line in file_1:
            seq = line.split(':')[0].strip()
            header = line.split(':')[1]
            seq_count.append(int(header.split('_')[0]))
        # Fix: Check if there are enough elements after splitting
        header_parts = header.split('_')
        max_key_header = header_parts[1] if len(header_parts) > 1 else "unknown"
        #print(max_key_header)
        num = sum(seq_count)
        #print(num)
        #print('break')
        seq_list.append(seq)
        header_list.extend([num, max_key_header])
        #print(header_list)
        final_header = ""
        for i, item in enumerate(header_list):
            final_header +=  str(item).rstrip("\n")+"_" 
        final_header_list.append(final_header)
        #print(final_header_list)
    return calculate_con_total(seq_list,final_header_list)

if __name__ == "__main__":
    start_time = time.time()
    
    
    input_dir = '/mnt/storage-HDD05a/1.scrach/immuno-mahek/WS1-scrach/crp/result/'
    file_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    num_cores = 8
    with Pool(processes=num_cores) as pool:
        results = pool.map(process_file, file_list)
        print(results)
    with open('sheep_light_con_file.txt', 'a') as file_con:
        for result in results:
            for c in result:
                file_con.write(f'{c}\n')
    end_time = time.time() 
    print(f"Script started at: {time.ctime(start_time)}")
    print(f"Script ended at: {time.ctime(end_time)}")
