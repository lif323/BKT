from Bayes_knowledge_tracing import BKT
import pandas as pd
import pickle

def load_from_file(data_path, min_problem_len):
    stu_id_dict = list()
    skill_col = list()
    res_col = list()
    order_col = list()
    stu_id_col = list()
    with open(data_path, "r") as f:
        lines = f.readlines()
    for i in range(0, len(lines), 4):
        seq_len, stu_id = list(map(int, lines[i].split(',')))
        if seq_len <= min_problem_len:
            continue
        stu_id_dict.append(stu_id)

        skills = list(map(int, lines[i+1].split(',')))
        res = list(map(int, lines[i+3].split(',')))
        orders = list(range(seq_len))
        stu_ids = list([stu_id ] * seq_len)

        skill_col += skills
        res_col += res
        order_col += orders
        stu_id_col += stu_ids
    
    return stu_id_dict, skill_col, res_col, order_col, stu_id_col

def read_data_to_csv(src_train_path, src_test_path, dst_path):

    min_problem_len = 20
    
    train_ids, train_skill_col, train_res_col, train_order_col, train_stu_id_col = load_from_file(src_train_path, min_problem_len=min_problem_len)
    test_ids, test_skill_col, test_res_col, test_order_col, test_stu_id_col = load_from_file(src_test_path, min_problem_len=min_problem_len)

    skill_col = train_skill_col + test_skill_col
    res_col = train_res_col + test_res_col
    order_col = train_order_col + test_order_col
    stu_id_col = train_stu_id_col + test_stu_id_col

    data = pd.DataFrame({'stu': stu_id_col, 'skills': skill_col, 'corrects': res_col, 'order': order_col}).astype(int)

    data.to_csv(dst_path, index=False)
    return train_ids

def get_bkt_data(data_path):
    data = pd.read_csv(data_path)
    skill_id_index = list(data['skills'].unique())
    bkt_data = {}
    for sk in skill_id_index:
        sk_data = data[data['skills'] == sk].sort_values('stu', ascending=True)
        stu_id_index = list(sk_data['stu'].unique())
        stu_dict = {}
        for stu in stu_id_index:
            stu_sk_data = sk_data[sk_data['stu'] == stu].sort_values('order', ascending=True)
            stu_dict[stu] = list(stu_sk_data['corrects'])
        bkt_data[sk] = stu_dict
    
    return bkt_data, max(skill_id_index)

def get_dkt_data(data_path):
    data = pd.read_csv(data_path)
    stu_index = list(data['stu'].unique())
    dkt_skill_dict = {}
    dkt_res_dict = {}
    for stu in stu_index:
        stu_df = data[data['stu'] == stu].sort_values('order', ascending=True)
        dkt_skill_dict[int(stu)] = list(stu_df['skills'])
        dkt_res_dict[int(stu)] = list(stu_df['corrects'])
    return dkt_skill_dict, dkt_res_dict

def run_bkt(data_path, train_ids):
    bkt_data, max_skill_id = get_bkt_data(data_path)
    dkt_skill, dkt_res = get_dkt_data(data_path)
    print("bkt, data over!!!!!!!!!!")
    DL, DT, DG, DS = {}, {}, {}, {}
    # i is the skill ID
    for i in bkt_data.keys():
        skill_data = bkt_data[i]
        train_data = []
        # Process in the order of students, j is the student ID, and filter out the ID belonging to the training set
        for j in skill_data.keys():
            if int(j) in train_ids:
                train_data.append(list(map(int, skill_data[j])))

        bkt = BKT(step=0.1, bounded=False, best_k0=True)

        # Delete the knowledge points with fewer practicing students and use the default parameters instead
        if len(train_data) > 2:
            DL[i], DT[i], DG[i], DS[i] = bkt.fit(train_data)
        else:
            DL[i], DT[i], DG[i], DS[i] = 0.5, 0.2, 0.1, 0.1
        
    mastery = bkt.inter_predict(dkt_skill, dkt_res, DL, DT, DG, DS, max_skill_id)

    return mastery

    
if __name__ == "__main__":
    train_data_path = './data/Ass_09_train.csv'
    test_data_path = './data/Ass_09_test.csv'
    dst_path = "data.csv"
    train_ids = read_data_to_csv(train_data_path, test_data_path, dst_path)
    print("to csv over!!!!")
    mastery = run_bkt(dst_path, train_ids)

    with open('mastery.pkl', 'wb') as f:
        pickle.dump(mastery, f)