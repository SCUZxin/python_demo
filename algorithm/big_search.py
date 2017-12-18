# 二分查找
# list.index()无法应对大规模数据的查询，需要用其它方法解决，这里谈的就是二分查找

def binarySearch(lst, value, low, high):
    if high < low:
        return -1
    mid = (low + high) // 2
    if lst[mid] > value:
        return binarySearch(lst, value, low, mid-1)
    elif lst[mid] < value:
        return binarySearch(lst, value, mid+1, high)
    else:
        return mid

#也可以不用递归方法，而采用循环，如下：
def bsearch(lst, value):
    lo, hi = 0, len(lst)-1
    while lo <= hi:
        mid = (lo + hi) // 2
        if lst[mid] < value:
            lo = mid + 1
        elif lst[mid] > value:
            hi = mid
        else:
            return mid
    return -1

if __name__ == "__main__":
    lst = range(3, 50)
    print(binarySearch(lst, 10, 0, len(lst)-1))
    print(bsearch(lst, 10))