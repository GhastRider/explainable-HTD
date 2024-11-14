import re

dicTrojan = {}
if 1:
    dicTrojan['RS232-T1000.v'] = ['U293', 'U294', 'U295', 'U296', 'U297', 'U298', 'U299',
                                'U300', 'U301', 'U302', 'U303', 'U304', 'U305']
    dicTrojan['RS232-T1100.v'] = ['U293', 'U294', 'U295', 'U296', 'U297', 'U298', 'U299',
                                'U300', 'U301', 'U302', 'U305', 'iDatasend_reg_2_']
    dicTrojan['RS232-T1200.v'] = ['U292', 'U293', 'U294', 'U295', 'U296', 'U297',
                                'U300', 'U301', 'U302', 'U303',
                                'iDatasend_reg_1', 'iDatasend_reg_2', 'iDatasend_reg_3', 'iDatasend_reg_4']
    dicTrojan['RS232-T1300.v'] = ['U292', 'U293', 'U294', 'U295', 'U296', 'U297',
                                'U302', 'U303', 'U304']
    dicTrojan['RS232-T1400.v'] = ['U292', 'U293', 'U294', 'U295', 'U296', 'U297', 'U298', 'U299',
                                'U300', 'U301', 'U302', 'U303', 'iDatasend_reg']
    dicTrojan['RS232-T1500.v'] = ['U293', 'U294', 'U295', 'U296', 'U297', 'U298', 'U299',
                                'U300', 'U301', 'U302', 'U303', 'U304', 'U305', 'iDatasend_reg_2_']
    dicTrojan['RS232-T1600.v'] = ['U293', 'U294', 'U295', 'U296', 'U297',
                                'U300', 'U301', 'U302', 'U303', 'U304',
                                'iDatasend_reg_1', 'iDatasend_reg_2']
    dicTrojan['RS232-T1700.v'] = ['U292', 'U293', 'U294', 'U295', 'U296', 'U297', 'U302', 'U303' ]
    dicTrojan['RS232-T1800.v'] = ['U300', 'U301', 'U302', 'U303' ]
    dicTrojan['RS232-T1900.v'] = ['U292', 'U293', 'U294', 'U296',
                                  'U303', 'U304', 'U305', 'U306', 'U307', 'U308', 'U309', 'U400',
                                  'iDatasend_reg_1', 'iDatasend_reg_2', 'iDatasend_reg_3', 'iDatasend_reg_4']
    dicTrojan['RS232-T2000.v'] = ['U292', 'U293', 'U294', 'U295', 'U296',
                                  'U302', 'U303', 'U308', 'U309', 'U401',
                                  'iDatasend_reg',]

    dicTrojan['s38584-T100.v'] = ['Trojan', 'NOT_test_se'] # 9
    dicTrojan['s15850-T100.v'] = ['Trojan', 'INVtest_se', 'Tg'] # 15
    dicTrojan['s38417-T100.v'] = ['Trojan'] # 12,15,15
    dicTrojan['s38417-T200.v'] = ['Trojan'] # 12,15,15
    dicTrojan['s38417-T300.v'] = ['Trojan', 'Ring'] # 12,15,15
    dicTrojan['s35932-T100.v'] = ['Trojan', 'INV_test_se'] # 15
    dicTrojan['s35932-T200.v'] = ['Trojan', 'INVtest_se', 'U5548', 'U5566', 'U6740', 'U6802'] # 16
    dicTrojan['s35932-T300.v'] = ['Tj', 'Trojan', 'INVtest_se'] # 15
    dicTrojan['wb-T100.v'] = ['Trojan'] # 15



file = [
        'RS232-T1000.v', 'RS232-T1100.v', 'RS232-T1200.v', 'RS232-T1300.v',
        'RS232-T1400.v', 'RS232-T1500.v', 'RS232-T1600.v', 'RS232-T1700.v',
        'RS232-T1800.v', 'RS232-T1900.v', 'RS232-T2000.v',

        's38417-T100.v', 's38417-T200.v', 's38417-T300.v',
        's35932-T100.v', 's35932-T200.v', 's35932-T300.v',
        's38584-T100.v', 's15850-T100.v',

        'c2670-T000.v', 'c2670-T050.v', 'c3540-T000.v', 'c3540-T050.v',
        'c5315-T000.v', 'c5315-T050.v', 'c6288-T000.v', 'c6288-T050.v',
        's1423-T000.v', 's1423-T200.v', 's13207-T000.v', 's13207-T200.v',
        's15850-T000.v', 's15850-T200.v', 's35932-T000.v', 's35932-T200.v',

        's1423-T400.v', 's1423-T600.v', 's13207-T400.v', 's13207-T600.v',
        's15850-T400.v', 's15850-T600.v', 's35932-T400.v', 's35932-T600.v',

        'AES-T100.v', 'AES-T200.v', 'AES-T300.v', 'AES-T400.v', 'AES-T500.v',
        'AES-T600.v', 'AES-T700.v', 'AES-T800.v', 'AES-T900.v', 'AES-T1000.v',
        'AES-T1100.v', 'AES-T1200.v', 'AES-T1300.v', 'AES-T1400.v', 'AES-T1500.v',
        'AES-T1600.v', 'AES-T1700.v', 'AES-T1800.v', 'AES-T1900.v', 'AES-T2000.v',
        'AES-T2100.v',

       ]


def find_logit_port(item): # 找出该例化模块的端口列表，并赋值给该节点
    port_list=[]
    item = re.search('\([\s\S]*\)', item).group()
    item = item.strip('(')
    item = item.strip(')')
    item = item.strip()
    for item1 in re.split(',',item):
        item1 = item1.strip()

        # 提取括号内部的变量
        item2 = item1.strip()
        if re.search('\(', item2) is not None:
            index = item2.find('(')
            item2 = item2[index + 1:]
        if re.search('\)', item2) is not None:
            index = item2.find(')')
            item2 = item2[:index]
        port_list.append(item2)

    return port_list

class Edge():
    countE = 0
    def __init__(self, name='', head='NULL',tail='NULL'):
        self.name = name
        self.head = head
        self.tail = tail
        Edge.countE += 1


class Queue():

    def __init__(self):
        self.front=0
        self.back=-1
        self.length=0
        self.list=[]

    def empty(self):
        if self.length<=0:
            return 1
        else:
            return 0

    def Print(self):
        for i in range(self.front, self.back+1):
            print(self.list[i])

    def push(self,ele):
        self.list.append(ele)
        self.length+=1
        self.back+=1

    def pop(self):
        if self.length>0:
            self.length -= 1
            self.front += 1
        else:
            print('队列为空，不能出队列')

    def top(self):
        return self.list[self.front]


def SplitofAssign(arg, width):
    # 去掉{}号，先分开逗号，然后分[]包含的多变量，若是[]是单变量，直接去掉空格输出即可

    res = []
    l1 = []
    arg = arg.strip()
    if re.search('\{', arg) is not None:
        arg = arg[1:-1].strip()

    # 将传进来的信息传到l1列表里
    if re.search(',', arg) is not None:
        l1 = arg.split(',')
    else:
        l1.append(arg.strip())

    for item in l1:
        item = item.strip()

        # []的多变量和单变量
        if re.search('\[', item) is not None:
            if re.search(':', item) is None :
                if re.search(' ', item) is not None:
                    tmp1 = item.split(' ')[0].strip()
                    tmp2 = item.split(' ')[1].strip()
                    item = tmp1 + tmp2
                res.append(item)
                # print('没有冒号，单独变量：', tmp1 + tmp2)
            else:
                index1 = item.find('[')
                index2 = item.find(']')
                tmpS = item[index1 + 1:index2]
                beg = int(tmpS.split(':')[0].strip())
                end = int(tmpS.split(':')[1].strip())
                if beg > end:
                    for i in range(beg, end-1, -1):
                        name = item[:index1].strip()
                        res.append(name + '[' + str(i) + ']')
                else:
                    for i in range(beg, end+1, 1):
                        name = item[:index1].strip()
                        res.append(name + '[' + str(i) + ']')
        else:
            # 单独变量且没有[]，直接插入结果
            if re.search('\'', item) is not None: # 如果是常数，就直接插入，因为到那边过去肯定直接去掉此句assign
                index = item.find('\'')
                num = int(item[:index])
                for i in range(num):
                    res.append(item)
            elif item not in width.keys():
                res.append(item)
            else:
                # print('是多变量但是只写了个名字')
                leng = width[item]
                for i in range(leng-1,-1,-1):
                    s = item + '[' + str(i) + ']'
                    res.append(s)
    return res


def calWidth(arg):
    arg = arg.strip()[1:].strip()
    tmplist = arg.split(':')
    if len(tmplist) != 2:
        print('当前信号长度判断有误')
        exit()
    else:
        big = max(int(tmplist[0]), int(tmplist[1]))
        small = min(int(tmplist[0]), int(tmplist[1]))
        return big - small +1
    return -1


def EdgeList(port):
    index = port.find('(')
    item = port[index+1:-1].strip()
    signal = ''

    if re.search('\'b', item) is not None:  # 如果是常数，就直接丢掉，因为不参与电路连接
        # print('1111当前判断是常数')
        signal = item
        # pass
    else:
        if re.search('\[', item) is not None:
            if re.search(' ', item) is not None:
                tmp1 = item.split(' ')[0].strip()
                tmp2 = item.split(' ')[1].strip()
                item = tmp1 + tmp2
            signal = item
        else:
            signal = item

    return signal







