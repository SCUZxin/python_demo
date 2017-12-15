# 导入 matplotlib 的所有内容（nympy 可以用 np 这个名字来使用）
from pylab import *

# 创建一个 8 * 6 点（point）的图，并设置分辨率为 80
figure(figsize=(8,6), dpi=80)

# 创建一个新的 1 * 1 的子图，接下来的图样绘制在其中的第 1 块（也是唯一的一块）
subplot(1, 1, 1)

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C, S = np.cos(X), np.sin(X)

# 添加图例并绘制曲线
# 绘制余弦曲线，使用蓝色的、连续的、宽度为 1 （像素）的线条
plot(X, C, color="blue", linewidth=2.0, linestyle="--", label='cosine')
# 绘制正弦曲线，使用绿色的、连续的、宽度为 1 （像素）的线条
plot(X, S, color="red", linewidth=2.0, linestyle="-", label='sine')
legend(loc='upper left')


# 设置横轴的上下限
xlim(X.min()*1.2, X.max()*1.21)

# 设置纵轴的上下限
ylim(C.min()*1.3, C.max()*1.3)

# 设置横轴记号
xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

# 设置纵轴记号
yticks([-1, 0, +1])

# 以分辨率 72 来保存图片
# savefig("exercice_2.png",dpi=72)

# 移动脊柱
ax = gca()
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('None')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

# 给一些特殊点(2π/3)做注释. 首先，我们在对应的函数图像位置上画一个点；
# 然后，向横轴引一条垂线，以虚线标记；最后，写上标签。
t = 2 * np.pi / 3
plot([t, t], [0, np.cos(t)], color='blue', linewidth=2.5, linestyle='--') #绘制垂线
scatter([t, ], [np.cos(t), ], 50, color='blue') # 绘制点，50是点的大小
annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',
         xy=(t, np.cos(t)), xycoords='data',
         xytext=(-90, -50), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))

plot([t, t], [0, np.sin(t)], color='red', linewidth='2.5', linestyle='--')
scatter([t, ], [np.sin(t), ], 50, color='red')
annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
         xy=(t, np.sin(t)), xycoords='data',
         xytext=(10, 30), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))

# 透明曲线，使记号标签能看见
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))

# 在屏幕上显示
show()