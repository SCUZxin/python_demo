#问题：
# 以人民币的硬币为例，假设硬币数量足够多。要求将一定数额的钱兑换成硬币。要求兑换硬币数量最少。

'''思路说明：
这是用贪婪算法的典型应用。在本例中用python来实现，主要思想是将货币金额除以某硬币单位，
然后去整数，即为该硬币的个数；余数则做为向下循环计算的货币金额。
这个算法的问题在于，得出来的结果不一定是最有结果。比如，硬币单位是[1,4,6],如果将8兑换成硬币，
按照硬币数量最少原则，应该兑换成为两个4（单位）的硬币，但是，按照本算法，得到的结果是一个6单位
和两个1单位的硬币。这也是本算法的局限所在。所谓贪婪算法，本质就是只要找出一个结果，不考虑以后会怎么样。
'''

#解决2(Python)
# 以下方法，以动态方式，提供最小的硬币数量。避免了贪婪方法的问题。
# 动态规划，找到状态，状态转移方程
# minCoins[i][j]:i分钱至少需要minCoins[i][j]个硬币，能使用的最高硬币为coinValues[i](很可能没用)

def coinChange(money, coinValues):
    minCoins = [[0 for j in range(money + 1)] for i in range(len(coinValues))]# [7][9]
    minCoins[0] = range(money + 1)    # 0---8，用coinValue[0]=1凑足0--8的钱需要0--8个硬币

    for i in range(1, len(coinValues)):     # 1--6
        for j in range(0, money + 1): # 0--8
            if j < coinValues[i]:
                minCoins[i][j] = minCoins[i - 1][j]
            else:
                minCoins[i][j] = min(minCoins[i - 1][j], 1 + minCoins[i][j - coinValues[i]])

    return minCoins[-1][-1]     # 返回需要的硬币的最少数目

if __name__ == "__main__":
    coin = [1, 2, 5, 10, 20, 50, 100]
    print(coinChange(8, coin))


