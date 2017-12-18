"""
#问题
二分查找
list.index()无法应对大规模数据的查询，需要用其它方法解决，这里谈的就是二分查找
#思路说明
在查找方面，python中有list.index()的方法。例如：
    >>> a=[2,4,1,9,3]           #list可以是无序，也可以是有序
    >>> a.index(4)              #找到后返回该值在list中的位置
    1
    >>> a.index(5)              #如果没有该值，则报错
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: 5 is not in list
这是python中基本的查找方法，虽然简单，但是，如果由于其时间复杂度为O(n)，对于大规模的查询恐怕是不足以胜任的。二分查找就是一种替代方法。
二分查找的对象是：有序数组。这点特别需要注意。要把数组排好序先。怎么排序，可以参看我这里多篇排序问题的文章。
基本步骤：
1. 从数组的中间元素开始，如果中间元素正好是要查找的元素，则搜素过程结束；
2. 如果某一特定元素大于或者小于中间元素，则在数组大于或小于中间元素的那一半中查找，而且跟开始一样从中间元素开始比较。
3. 如果在某一步骤数组为空，则代表找不到。
这种搜索算法每一次比较都使搜索范围缩小一半。时间复杂度：O(logn)
"""

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