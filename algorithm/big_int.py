# 问题：大整数相乘
# Python支持“无限精度”的整数，一般不用考虑溢出的问题
# 其他语言，通常采取“分治法”解决大整数相乘问题


def arabic_multiplication(num1, num2):
    num_list1 = [int(i) for i in str(num1)] # 将int 类型的123转化为list类型的[1, 2, 3], 每个元素都是int
    num_list2 = [int(i) for i in str(num2)]

    # 两个list中的元素两两相乘得到矩阵,list类型
    int_matrix = [[i*j for i in num_list1] for j in num_list2]
    # 将list的元素由 int 转化为 str，主要是 9 ->'09'
    str_matrix = [list(map(convert_to_str, int_matrix[i])) for i in range(len(int_matrix))]
    # print(int_matrix)
    # print(str_matrix)
    # print(len(num_list1))
    # print(len(num_list2))

    length = len(num_list1+num_list2)
    hypotenuse_matrix = [[] for i in range(length)] # 通过for循环定义一个二维list

    # 循环将str_matrix中的元素按照斜边赋值到二维list中
    for i in range(len(num_list2)):
        for j in range(len(num_list1)):
            x_top_left = int(str_matrix[i][j][0])
            x_bottom_right = int(str_matrix[i][j][1])
            hypotenuse_matrix[i+j].append(x_top_left)
            hypotenuse_matrix[i+j+1].append(x_bottom_right)

    sum = 0
    carry = 0    # 进位
    # 记录上述各位数字相应乘积的十位数与个位数，把这些乘积由右到左，沿斜线方向相加，相加满十时要向前进一。最后得到两数乘积
    for i in range(len(hypotenuse_matrix)):
        index_now = length-1-i
        subtotal = list_sum(hypotenuse_matrix[index_now]) + carry

        str_subtotal = convert_to_str(subtotal)
        digits = int(str_subtotal[1])   # 个位
        carry = int(str_subtotal[0])    # 十位，进位

        sum += digits*(10**(i))

    return sum


# 将int类型转化为str类型，9-->'09'
def convert_to_str(num):
    if num < 10:
        return '0' + str(num)
    else:
        return str(num)


def list_sum(list_temp):
    sum_temp = 0
    for i in range(len(list_temp)):
        sum_temp += list_temp[i]
    return sum_temp

if __name__ == "__main__":
    num1, num2 = 1234567890123, 789456123
    product = arabic_multiplication(num1, num2)
    print(product)
    print(num1 * num2)

