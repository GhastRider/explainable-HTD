

def show_nxgraph_info(G):
    # 获取有向图的所有边的信息
    all_edges = G.edges(data=True)

    print("number_of_nodes()", G.number_of_nodes())
    print("number_of_edges()", G.number_of_edges())

    nxdict = []
    # 打印边的信息
    for edgeingraph in all_edges:
        # print(f"边: {edge[0]} -> {edge[1]}, 属性: {edge[2]}")
        nxdict.append([edgeingraph[0], edgeingraph[1]])
    print("networkx有向边对象中边的列表集合：", len(nxdict), nxdict)


def counting_edge(my_dict):
    # 示例字典，每个键对应若干个列表
    # my_dict = {'a': [[1, 2, 3], [4, 5]], 'b': [[6, 7], [8, 9, 10]], 'c': [[11, 12], [3, 2, 1]]}

    # 获取字典中的所有列表
    all_sets = [set(tuple(sublist)) for lists in my_dict.values() for sublist in lists]

    # 获取字典中的所有列表，并将每个列表转换为集合
    all_sets = [set(tuple(sublist)) for lists in my_dict.values() for sublist in lists]

    # 将所有集合转化为集合（去重）
    unique_sets = set(tuple(s) for s in all_sets)

    # 统计集合的个数
    unique_sets_count = len(unique_sets)

    print(len(all_sets))
    # 显示结果
    print(f"不计相同元素的集合个数：{unique_sets_count}")
    print(f"集合1：{unique_sets}\n")



def deduplication_counting(my_dict):
    # my_dict = {'a': [[1, 2, 3], [4, 5]], 'b': [[6, 7], [8, 9, 10]], 'c': [[11, 12], [3, 2, 1]]}

    # 将每个子列表的元素合并
    combined_sublists = [list(set(sublist)) for sublist_list in my_dict.values() for sublist in sublist_list]
    # print(len(combined_sublists))

    # 去重
    unique_result = list(map(list, set(map(tuple, combined_sublists))))
    print("deduplication_counting results去重后", len(unique_result))

    # 统计含有NULL的子列表
    null_sublists = [sublist for sublist in unique_result if 'NULL' in sublist]
    print("包含 'NULL' 的子列表：", len(null_sublists), null_sublists)

    # 统计并去除包含 'NULL' 的子列表
    filtered_list = [sublist for sublist in unique_result if 'NULL' not in sublist]
    print("去重并除NULL后edge数量：", len(filtered_list))

    return filtered_list