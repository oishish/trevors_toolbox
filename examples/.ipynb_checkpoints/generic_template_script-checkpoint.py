import numpy as np
import matplotlib.pyplot as plt
import trevorarp as tb


iden = "identifier"
#data = tb.data.get_dv_data(iden)

# Random test data
np.random.seed(42)
data2 = np.zeros((100,100))
rows, cols = data2.shape
for i in range(rows):
    data2[i,:] = 0.25*np.random.rand(100) + np.linspace(0,1,100)

xinches = 9.5
yinches = 4.7
fi = tb.display.figure_inches(iden+"_", xinches, yinches, dark=True)
xmargin = 0.8
ymargin = 0.7

height = 3.25
width = 4.0

xint = 0.75

ax1 = fi.make_axes([xmargin, ymargin, width, height])
ax2, ax2cb = fi.make_img_axes([xmargin+xint+width, ymargin, height, height])

fi.stamp(iden)

# ax1.plot(data1)
for i in range(3):
    ax1.plot(data2[i,:]+i)
ax1.set_xlabel("X Axis Label")
ax1.set_ylabel("Y Axis Label")
ax1.set_title("Title")

cmap, cnorm, smap = tb.display.colorscale_map([0,1.25], cmin=0.0, cmax=1.0, mapname='inferno')
tb.display.make_colorbar(ax2cb, cmap, cnorm)

ax2.imshow(data2, cmap=cmap, norm=cnorm)
ax2.set_xlabel("X Axis Label")
ax2.set_ylabel("Y Axis Label")
ax2.set_title("Title", loc='left')

plt.show()
