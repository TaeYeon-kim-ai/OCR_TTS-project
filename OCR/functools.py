def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


#===========================================
def xy_compare(x, y):
    if x[1] > y[1]: # y좌표가 작은 것부터 앞으로
        return 1
    elif x[1] == y[1]: # y좌표가 같을 경우
        if x[0] > y[0]: # x 좌표가 작은 것이 앞으로 나오게
            return 1
        elif x[0] < y[0]: # x 좌표가 큰 것이 뒤로
            return -1
        else: # 같은 경우에는 그대로
            return 0
    else: # y좌표가 큰 것이 뒤로
        return -1