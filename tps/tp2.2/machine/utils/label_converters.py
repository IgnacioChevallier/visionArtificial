def label_to_int(string_label):
    if string_label == 'spade': return 1
    if string_label == 'heart': return 2
    if string_label == 'diamond': return 3
    if string_label == 'club': return 4
    else:
        raise Exception('unknown class_label: ' + string_label)

def int_to_label(int_label):
    if int_label == 1: return 'spade'
    if int_label == 2: return 'heart'
    if int_label == 3: return 'diamond'
    if int_label == 4: return 'club'
    else:
        raise Exception('unknown int_label: ' + str(int_label))